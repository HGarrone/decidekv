"""
Embedding catalogue for option sets too large to keep in the KV cache.

`TopKIndex` asks the catalogue for the k options closest to the state, writes only these in the prompt (best first)
and lets the LLM choose among them. This is the standard retrieve-then-rerank pipeline; nothing is cached or stored.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from .core import Index, parse_options, render_state
from .primitives import ChoiceAnswer

QUERY_TEMPLATE = "Instruct: {question}\nQuery:{state}"


class OllamaEncoder:
    """Embeddings from an Ollama embedding model (e.g. qwen3-embedding:4b), L2-normalised."""

    def __init__(self, model: str = "qwen3-embedding:4b", host: str | None = None):
        from .ollama import Ollama
        self.client = Ollama(model, host)

    def __call__(self, texts, batch: int = 64):
        out = []
        for i in range(0, len(texts), batch):
            out += self.client.embed(list(texts[i:i + batch]))
        e = np.asarray(out, dtype=np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


class Catalog:
    """Option vectors and the query-to-options search. `head` is an optional trained projection
    (see `kvindex.train`) applied to option and query vectors."""

    def __init__(self, texts, encoder, vectors=None, head=None, query_template: str = QUERY_TEMPLATE):
        self.encoder, self.query_template = encoder, query_template
        self.head = head
        self.raw = np.asarray(vectors if vectors is not None else encoder(list(texts)), dtype=np.float32)
        self.vec = self.head.options(self.raw) if self.head is not None else self.raw

    @staticmethod
    def cached(texts, encoder, folder: str):
        """Vectors of `texts`, computed by `encoder` once and saved in `folder` (keyed by the content of the texts
        and by the class name of the encoder); later calls read them from disk."""
        key = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
        path = os.path.join(folder, f"catalog_{type(encoder).__name__}_{key}.npy")
        if os.path.exists(path):
            return np.load(path)
        vec = np.asarray(encoder(list(texts)), dtype=np.float32)
        os.makedirs(folder, exist_ok=True)
        np.save(path, vec)
        return vec

    def query(self, state, question: str) -> np.ndarray:
        v = self.query_raw(state, question)
        if self.head is not None:
            return self.head.state(v[None])[0]
        return v

    def top(self, v: np.ndarray, k: int, banned: set[int] = frozenset(), raw: bool = False) -> list[int]:
        """Positions of the k options closest to the query vector v (raw=True: untrained option vectors)."""
        s = (self.raw if raw else self.vec) @ v
        if banned:
            s = s.copy()
            s[list(banned)] = -np.inf
        k = min(k, len(s) - len(banned))
        idx = np.argpartition(-s, k - 1)[:k]
        return [int(i) for i in idx[np.argsort(-s[idx])]]

    def query_raw(self, state, question: str) -> np.ndarray:
        """Embedding of the query, without the head."""
        v = self.encoder([self.query_template.format(question=question, state=render_state(state))])[0]
        return v


class TopKIndex:
    """Index for large option sets: the catalogue selects the `k` options closest to the state and the
    LLM chooses among them. `last_candidates` holds the options shown to the LLM in the last decision."""

    def __init__(self, kv, options, encoder, k: int = 50, vectors=None, head=None, task: str | None = None,
                 question: str | None = None, noun: str = "option", state_label: str = "State",
                 query_template: str | None = None, fallback: int = 5):
        self.kv, self.k = kv, k
        self.keys, self.descs, texts = parse_options(options)
        self.n = len(self.keys)
        self.pos = {name: i for i, name in enumerate(self.keys)}
        self.prompt = dict(task=task, question=question, noun=noun, state_label=state_label)
        self.catalog = Catalog(texts, encoder, vectors, head, query_template or QUERY_TEMPLATE)
        self.fallback = fallback
        self.fallbacks = 0
        trained = head.report.get("trained_options") if head is not None else None
        self.untrained = set(range(self.n)) - set(trained) if trained is not None else set()

    def choose(self, state, instructions=None, usage=None, exclude=()) -> ChoiceAnswer:
        banned = set(self.pos[e] for e in exclude if e in self.pos)
        q = instructions or self.prompt["question"] or ""
        k = min(self.k, len(self.keys) - len(banned))
        if k == 0:
            raise ValueError("every option is excluded")
        v_raw = self.catalog.query_raw(state, q)
        head = self.catalog.head
        v = head.state(v_raw[None])[0] if head is not None else v_raw
        top = self.catalog.top(v, k, banned)
        if k == 1 and self.fallback and self.untrained:
            raw_top = self.catalog.top(v_raw, self.fallback, banned, raw=True)
            if raw_top[0] in self.untrained and raw_top[0] != top[0]:
                self.fallbacks += 1
                near = self.catalog.top(v, self.fallback, banned)
                top = near + [i for i in raw_top if i not in near]
        self.last_candidates = [self.keys[i] for i in top]
        if len(self.last_candidates) == 1:
            return ChoiceAnswer(choice=self.last_candidates[0], probabilities={}, confidence=1.0)
        sub = Index(self.kv, {self.keys[i]: self.descs[i] for i in top}, warm=False, **self.prompt)
        return sub.choose(state, instructions, usage)

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"TopKIndex(n={self.n}, k={self.k})"
