"""
kvindex core: options-first prompts whose KV cache acts as a reusable index.

Layout of every prompt:   [header + options]  [state]  [question]  -> answer
The options block is byte-identical across calls, so the inference server keeps its KV cache and only
the state/question tokens are computed. Questions on the same state also share [options][state].
"""
from __future__ import annotations

import difflib
import json
import math
import time
import warnings
from typing import Any, Iterable

from .ollama import HYBRID_ARCHS, Ollama
from .primitives import (Choice, ChoiceAnswer, Noul, NoulAnswer, Response, Score, ScoreAnswer, Usage,
                         answer_confidence, confidence, softmax)

LETTERS = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]   # 52 single-token labels
MAX_LABELS = len(LETTERS)
CHARS_PER_TOKEN = 3.0          # conservative estimate used to size num_ctx before the first call


def render_state(state: Any) -> str:
    """Strings pass through; dicts become `key: value` lines; lists become `- item` lines."""
    if isinstance(state, str):
        return state.strip()
    if isinstance(state, dict):
        return "\n".join(f"{k}: {render_state(v) if isinstance(v, (dict, list)) else v}" for k, v in state.items())
    if isinstance(state, (list, tuple)):
        return "\n".join(f"- {render_state(x)}" for x in state)
    return json.dumps(state, ensure_ascii=False)


def _ctx_for(chars: int) -> int:
    need = int(chars / CHARS_PER_TOKEN) + 512
    return max(2048, 1 << math.ceil(math.log2(need)))


def parse_options(options: Any) -> tuple[list[str], list[str], list[str]]:
    """Names, descriptions and prompt texts of a list of names or a {name: description}
    mapping. Names must be unique and at least two."""
    if isinstance(options, dict):
        keys = [str(k_) for k_ in options]
        descs = [str(d) if d not in (None, "") else "" for d in options.values()]
    else:
        keys = [str(o) for o in options]
        descs = [""] * len(keys)
    texts = [f"{k_}: {d}" if d else k_ for k_, d in zip(keys, descs)]
    if len(keys) < 2:
        raise ValueError("an Index needs at least 2 options")
    if len(set(keys)) != len(keys):
        raise ValueError("options must be unique")
    return keys, descs, texts


WORDING = {
    "intro": "You are a decision engine.{task} Answer the question about the {state} by choosing one {noun} from the "
             "list of {plural}. {reply}\n\n",
    "reply_name": "Reply with the {noun} exactly as written in the list{part}, and nothing else.",
    "reply_label": "Reply with the label of the {noun} (the text before the colon), and nothing else.",
    "part": " (only the part before the colon)",
    "plural": "{noun}s",
    "question": "Which {noun} best matches the {state}?",
    "question_title": "Question",
    "shared_intro": "You are a decision engine. Questions about the {state} below follow, one at a time.\n\n",
}


class Index:
    """A fixed set of options, pre-filled into the model's KV cache. Create it with `KV.index(...)`.

    The answer is the option itself, decoded under a constraint that only allows options of the list
    (grammar-constrained up to `enum_max` options on Ollama, a token trie with the llama.cpp backend).
    `labels=True` answers with single-token labels A-Z/a-z instead (one forward pass, a probability for
    every option, at most 52 options); it is less accurate when options are words, because the first
    letter of the intended word gets read as a label.

    Name things for the model: `noun` ("intent", "word", "tool"…), `state_label` ("Customer message",
    "Definition"…) and `question` become part of the cached prompt. A domain-specific prompt measured up
    to +16 points over a generic one.
    `wording` overrides the fixed words of the prompt (keys of `WORDING`): for prompts in another language.
    """

    def __init__(self, kv: "KV", options: Any, labels: Any = False, task: str | None = None,
                 question: str | None = None, noun: str = "option", state_label: str = "State",
                 enum_max: int = 500, warm: bool = True, wording: dict | None = None):
        self.kv = kv
        self.keys, self.descs, self.texts = parse_options(options)
        self.n = len(self.keys)
        self.task, self.question, self.noun, self.state_label = task, question, noun, state_label
        self.wording = {**WORDING, **(wording or {})}
        if isinstance(labels, (list, tuple)):
            self.labels = list(labels)
        elif labels:
            if self.n > MAX_LABELS:
                raise ValueError(f"labels=True supports at most {MAX_LABELS} options")
            self.labels = LETTERS[:self.n]
        else:
            self.labels = None
        self.enum_max = enum_max
        self._intro, self._options = self.intro(), self.options_block()
        kv.require_ctx(_ctx_for(len(self.prefix) + 2000))
        self.prompt_tokens = None
        if warm:
            self.warm()

    # ------------------------------------------------------------ prompt blocks
    # The same five blocks, with the same words, are used by every architecture; only their order and the way
    # the answer is read change. kvindex uses: [intro + options] (cached) | [state] | [question + answer head].
    def intro(self) -> str:
        noun, plural = self.noun, self.wording["plural"].format(noun=self.noun)
        task = f" {self.task.strip()}" if self.task else ""
        if self.labels:
            reply = self.wording["reply_label"].format(noun=noun)
        else:
            part = self.wording["part"] if self.texts != self.keys else ""
            reply = self.wording["reply_name"].format(noun=noun, part=part)
        return self.wording["intro"].format(task=task, state=self.state_label.lower(), noun=noun, plural=plural,
                                             reply=reply)

    def options_block(self) -> str:
        if self.labels:
            own = self.labels == self.keys          # e.g. Score levels "0".."n" or Noul "Yes"/"No"
            body = "\n".join(f"{lab}: {(d or k_) if own else t}"
                             for lab, k_, d, t in zip(self.labels, self.keys, self.descs, self.texts))
        else:
            body = "\n".join(f"- {t}" for t in self.texts)
        return f"### {self.wording['plural'].format(noun=self.noun).capitalize()}\n{body}\n\n"

    def state_block(self, state: Any) -> str:
        return f"### {self.state_label}\n{render_state(state)}\n\n"

    def question_block(self, instructions: str | None) -> str:
        q = instructions or self.question or self.wording["question"].format(noun=self.noun,
                                                                                 state=self.state_label.lower())
        return f"### {self.wording['question_title']}\n{q}\n\n"

    def answer_head(self) -> str:
        return f"### {self.noun.capitalize()}\n"

    @property
    def prefix(self) -> str:
        return self._intro + self._options

    def segments(self, state: Any, instructions: str | None, order: str = "options-first") -> tuple[str, str, str]:
        """options-first: [intro + options] [state] [question + head]; the first segment is shared by every
        call, so the backend keeps it in the KV cache. state-first: [intro + state] [question] [options + head].
        shared-state: [shared intro + state] [intro + options] [question + head]; the same first segment for
        every question about one state."""
        if order == "options-first":
            return (self.prefix, self.state_block(state), self.question_block(instructions) + self.answer_head())
        if order == "shared-state":
            head = self.wording["shared_intro"].format(state=self.state_label.lower())
            return (head + self.state_block(state), self.prefix, self.question_block(instructions) + self.answer_head())
        return (self._intro + self.state_block(state), self.question_block(instructions),
                self._options + self.answer_head())

    def prompt(self, state: Any, instructions: str | None) -> str:
        return "".join(self.segments(state, instructions))

    def warm(self) -> "Index":
        """Fill the KV cache with the options block (the offline 'indexing' step)."""
        self.kv._check_arch()
        st = self.kv.client.prefill(self.segments("(empty)", None), self.num_ctx)
        self.prompt_tokens = st["input_tokens"] + st["cached_tokens"]
        if self.prompt_tokens + 256 > self.num_ctx:
            self.kv.require_ctx(1 << math.ceil(math.log2(self.prompt_tokens + 1024)))
            self.warm()
        return self

    @property
    def num_ctx(self) -> int:
        return self.kv.num_ctx

    # ------------------------------------------------------------ answering
    def choose(self, state: Any, instructions: str | None = None, usage: Usage | None = None,
               exclude: Iterable[str] = (), full: bool = False, order: str = "options-first") -> ChoiceAnswer:
        """Pick the best option for `state` (one call to the LLM).

        `exclude` removes options from the choice (e.g. attempts already refuted by the environment in a
        trial-and-error loop): the decoding constraint makes them impossible to pick, not just discouraged.
        `full=True` scores every option (the log-probability of its tokens, normalised over the options) instead of
        decoding the best one greedily: `probabilities` then covers all the options. It costs one forward pass per
        distinct token prefix of the options, so use it with a few dozen options or when the distribution matters.

        `order` is the order of the prompt blocks (see `segments`); `KV.decide` uses "shared-state" for many
        questions about one state."""
        excluded = set(exclude)
        allowed = [k_ for k_ in self.keys if k_ not in excluded] if excluded else self.keys
        if not allowed:
            raise ValueError("every option is excluded")
        usage = usage if usage is not None else Usage()
        if full and not self.labels:
            return self._choose_scored(state, instructions, usage, allowed, order)
        if self.labels:
            return self._choose_labels(state, instructions, usage, allowed, order)
        return self._choose_name(state, instructions, usage, allowed, order)

    def scores(self, state: Any, instructions: str | None = None, allowed: Iterable[str] | None = None,
               usage: Usage | None = None, order: str = "options-first") -> dict[str, float]:
        """log P of every option for `state` (normalised over `allowed`, all options when None)."""
        allowed = list(allowed) if allowed is not None else self.keys
        lps, st = self.kv.client.score_options(
            self.segments(state, instructions, order),
            self.keys,
            allowed=allowed
        )
        self._account(st, usage if usage is not None else Usage())
        return lps

    def _choose_scored(self, state, instructions, usage, allowed, order="options-first") -> ChoiceAnswer:
        lps = self.scores(state, instructions, allowed, usage, order)
        probs = softmax(lps)
        full = {k_: round(probs.get(k_, 0.0), 6) for k_ in self.keys}
        best = max(probs, key=probs.get)
        return ChoiceAnswer(choice=best, probabilities=full, confidence=round(confidence(list(full.values())), 4))

    @staticmethod
    def _account(st: dict, usage: Usage):
        usage.input_tokens += st.get("input_tokens", 0)
        usage.cached_tokens += st.get("cached_tokens", 0)
        usage.output_tokens += st.get("output_tokens", 0)

    def _choose_labels(self, state, instructions, usage, allowed, order="options-first") -> ChoiceAnswer:
        segments, labels = self.label_request(state, instructions, allowed, order)
        lps, st = self.kv.client.label_logprobs(segments, labels, self.num_ctx)
        self._account(st, usage)
        return self.label_answer(lps, labels)

    def _choose_name(self, state, instructions, usage, allowed, order="options-first") -> ChoiceAnswer:
        text, lp, st = self.kv.client.pick_option(
            self.segments(state, instructions, order),
            self.keys,
            self.num_ctx,
            self.enum_max,
            allowed=None if allowed is self.keys else allowed
        )
        self._account(st, usage)
        best = self._match(text or "", allowed)
        p = round(math.exp(lp), 6) if lp is not None else None
        probs = {best: p} if p is not None else {}
        return ChoiceAnswer(choice=best, probabilities=probs, confidence=p if p is not None else 0.0)

    def _match(self, text: str, allowed: list[str] | None = None) -> str:
        """Map decoded text to an option (always one of `allowed`)."""
        pool = allowed if allowed is not None else self.keys
        t = text.strip().strip('"').strip()
        low = {k_.lower(): k_ for k_ in pool}
        if t.lower() in low:
            return low[t.lower()]
        head = t.split(":")[0].strip().lower()
        if head in low:
            return low[head]
        close = difflib.get_close_matches(t.lower(), list(low), n=1, cutoff=0.0)
        return low[close[0]] if close else pool[0]

    def __len__(self):
        return self.n

    def __repr__(self):
        mode = "labels+logprobs" if self.labels else ("enum" if self.n <= self.enum_max else "free+match")
        return f"Index(n={self.n}, mode={mode}, prompt_tokens={self.prompt_tokens})"

    def label_request(self, state, instructions, allowed, order="options-first"):
        """(segments, labels) of one label decision, to be answered by the backend."""
        by_key = dict(zip(self.keys, self.labels))
        labels = [by_key[k_] for k_ in allowed]
        return self.segments(state, instructions, order), labels

    def label_answer(self, lps, labels) -> ChoiceAnswer:
        """The answer from the label log-probabilities returned by the backend."""
        lp = {self.keys[self.labels.index(lab)]: v for lab, v in lps.items() if lab in labels}
        probs = softmax(lp)
        full = {k_: round(probs.get(k_, 0.0), 6) for k_ in self.keys}
        best = max(probs, key=probs.get)
        return ChoiceAnswer(choice=best, probabilities=full, confidence=round(confidence(list(full.values())), 4))


class KV:
    """Entry point. `KV("qwen3:8b")` loads any pure-attention GGUF model that Ollama has downloaded.

    >>> kv = KV("qwen3:8b")
    >>> intents = kv.index(["card arrival", "lost or stolen card", "refund not showing up"])
    >>> intents.choose("I still haven't received my new card").choice
    'card arrival'
    """

    def __init__(self, model: str, backend: str = "llamacpp", host: str | None = None, keep_alive: str = "30m",
                 embed: Any = None, flat_max: int = 3000, query_template: str | None = None, **backend_kw):
        """backend="llamacpp" (default: in-process llama.cpp; loads the GGUF that Ollama already downloaded, or a
        .gguf path; extra keyword arguments such as n_ctx and kv_type go to the backend) or "llamaserver" (a
        llama.cpp `llama-server`: with `host` the URL of a running server, without it kvindex starts one on the
        model; extra keyword arguments such as n_ctx, slots and port go to `LlamaServer.start`) or "ollama"
        (talks to the Ollama server).

        `embed` is the embedding model of the catalogue used for option sets larger than `flat_max` (a model name
        such as "qwen3-embedding:4b", or any callable that maps a list of texts to L2-normalised vectors).
        `query_template` is the text sent to the embedding model for a state (placeholders {state} and {question});
        the default suits instruction-tuned embedding models such as Qwen3-Embedding."""
        if backend == "ollama":
            self.client = Ollama(model, host, keep_alive)
        elif backend == "llamacpp":
            from .llamacpp import LlamaCpp
            self.client = LlamaCpp(model, **backend_kw)
        elif backend == "llamaserver":
            from .llamaserver import LlamaServer
            self.client = LlamaServer(host) if host else LlamaServer.start(model, **backend_kw)
        else:
            raise ValueError(f"unknown backend {backend!r} (use 'llamacpp', 'llamaserver' or 'ollama')")
        self.model = model
        self.backend = backend
        self.embed = embed
        self.flat_max = flat_max
        self.query_template = query_template
        # one context size for the whole session: Ollama reloads the model whenever num_ctx changes
        self.num_ctx = 4096
        self._indexes: dict[tuple, Any] = {}
        self._arch_checked = False
        self._encoder = None

    def require_ctx(self, n: int):
        """Grow (never shrink) the shared context size."""
        self.num_ctx = max(self.num_ctx, n)

    def close(self):
        """Free the models (the llama.cpp backend keeps GPU memory until it is closed)."""
        for obj in [self._encoder, self.client]:
            if obj is not None and hasattr(obj, "close"):
                obj.close()
        self._encoder = None
        self._indexes.clear()

    def _check_arch(self):
        if self._arch_checked:
            return
        self._arch_checked = True
        arch = self.client.architecture()
        if arch in HYBRID_ARCHS and self.backend == "ollama":
            warnings.warn(f"{self.model} uses a hybrid recurrent architecture ({arch}): the Ollama server cannot "
                          "reuse a partial prefix, so answers stay correct but each call recomputes the options. "
                          'Use backend="llamacpp" (resident prefixes work on every architecture).', stacklevel=3)

    @property
    def encoder(self):
        """The embedding model of the catalogue, loaded on first use."""
        if self._encoder is None:
            if self.embed is None:
                raise ValueError(f"more than {self.flat_max} options need an embedding model: "
                                 'KV(model, embed="qwen3-embedding:4b")')
            if callable(self.embed):
                self._encoder = self.embed
            elif self.backend == "llamacpp":
                from .llamacpp import LlamaCppEncoder
                self._encoder = LlamaCppEncoder(self.embed)
            else:
                from .catalog import OllamaEncoder
                self._encoder = OllamaEncoder(self.embed, self.client.host)
        return self._encoder

    def train(self, options: Any, examples: Any, question: str | None = None, out: str | None = None,
              **fit_kw) -> Any:
        """Train the embedding catalogue of `options` on `examples`, a list of (state, name of the right option).

        Only the catalogue is trained (a small projection of the embedding vectors, see `kvindex.train`); the LLM is
        not. Use the same `question` as in `KV.index`. The returned head goes to `KV.index(options, head=head)`;
        `head.report` holds the validation accuracy before and after training. `out` saves it to a .npz file."""
        from .catalog import QUERY_TEMPLATE
        from .train import fit_head
        keys, _, texts = parse_options(options)
        pos = {name: i for i, name in enumerate(keys)}
        examples = list(examples)
        for state, option in examples:
            if str(option) not in pos:
                raise ValueError(f"example option not in the option list: {option!r}")
        encoder = self.encoder
        option_vectors = encoder(texts)
        template = self.query_template or QUERY_TEMPLATE
        state_vectors = encoder([template.format(question=question or "", state=render_state(state))
                                 for state, _ in examples])
        head = fit_head(state_vectors, option_vectors, [pos[str(option)] for _, option in examples], **fit_kw)
        if out:
            head.save(out)
        return head

    def index(self, options: Any, task: str | None = None, question: str | None = None, noun: str = "option",
              state_label: str = "State", labels: Any = False, k: int | None = None, head: Any = None, **kw) -> Any:
        """Build an index over `options` (list of names or {name: description}).

        Up to `flat_max` options the whole list is written once in the KV cache and every decision computes only
        the state (`Index`). Above it the embedding catalogue selects the `k` closest options for each decision and
        the LLM chooses among them (`TopKIndex`).

        `head` is a trained catalogue (the object returned by `KV.train`, or the path of its .npz file). With a head
        the catalogue is used at every size and, unless `k` is given, it decides alone (k=1, no LLM call).

        A head trained with examples of only some options gets the others wrong. With `fallback=5` (the default)
        the LLM chooses among the 5 best options of the trained and of the untrained catalogue whenever the
        untrained catalogue prefers an option without examples; `fallback=0` switches it off. A head trained on
        every option never falls back.

        >>> kv.index(intents, noun="intent", state_label="Customer message",
        ...          question="Which intent does this customer message express?")
        """
        if k is None:
            k = 1 if head is not None else 50
        key = (tuple(options.items()) if isinstance(options, dict) else tuple(options), task, question, noun,
               state_label, tuple(labels) if isinstance(labels, (list, tuple)) else bool(labels), k,
               head if isinstance(head, str) else id(head) if head is not None else None,
               tuple(sorted((k_, repr(v)) for k_, v in kw.items())))
        if key not in self._indexes:
            if head is not None or len(options) > self.flat_max:
                if labels:
                    raise ValueError("labels are not available with the embedding catalogue")
                from .catalog import TopKIndex
                if isinstance(head, str):
                    from .train import Head
                    head = Head.load(head)
                self._indexes[key] = TopKIndex(self, options, self.encoder, k=k, head=head,
                                               query_template=self.query_template, task=task,
                                               question=question, noun=noun, state_label=state_label, **kw)
            else:
                self._indexes[key] = Index(self, options, labels=labels, task=task, question=question, noun=noun,
                                           state_label=state_label, **kw)
        return self._indexes[key]

    def choose(self, state: Any, options: Any, instructions: str | None = None) -> ChoiceAnswer:
        """One-shot convenience: the index for `options` is built on first use and reused afterwards."""
        if hasattr(options, "choose"):
            idx = options
        else:
            idx = self.index(options)
        return idx.choose(state, instructions)

    def decide(self, state: Any, questions: dict[str, Any] | Any = None, temperature: float = 1.0,
               labels: bool = False, full: bool = False, head: Any = None, order: str = "auto",
               threshold: float | None = None) -> Response:
        """OpenJev / Jev-style: several typed questions about one state.

        `questions` maps ids to `Choice`, `Score` or `Noul` (these classes, OpenJev's, or plain dicts with a
        "type" field). An OpenJev `SystemOneRequest` can be passed as the only argument.
        `full=True` gives `Choice` answers the probability of every option (see `Index.choose`); `Score` and `Noul`
        answers always carry the probability of every level.
        `head` (from `KV.train_questions`, or the path of its .npz file) answers every question with the trained
        catalogue alone: one embedding call per state, no LLM call.
        `threshold` with a head makes a cascade: the catalogue answers every question, and the questions whose
        answer has a confidence below `threshold` (1 - normalised entropy, as in OpenJev) are asked to the LLM in
        one call; `Response.escalated` lists them.
        `order`: "options" keeps every option list first in the prompt (cached across states); "state" puts the
        state first, shared by all the questions of the call; "auto" (default) puts the state first for the
        questions with a short option list (at most 2,000 characters, or shorter than the state), when there
        are at least two of them, and keeps long lists first (cached across states)."""
        if questions is None and hasattr(state, "questions"):
            state, questions = state.state, state.questions
        if head is not None:
            if isinstance(head, str):
                from .train import Head
                head = Head.load(head)
            cat = self._decide_catalog(state, questions, head)
            if threshold is None:
                return cat
            low = [qid for qid, a in cat.answers.items() if answer_confidence(a) < threshold]
            if not low:
                return cat
            llm = self.decide(state, {qid: questions[qid] for qid in low}, temperature=temperature, labels=labels,
                              full=full, order=order)
            answers = {qid: llm.answers.get(qid, a) for qid, a in cat.answers.items()}
            return Response(model=f"kvindex/{self.model}/cascade", answers=answers, usage=llm.usage,
                            latency_ms=round(cat.latency_ms + llm.latency_ms, 2), escalated=low)
        t0 = time.perf_counter()
        usage, answers = Usage(), {}
        items = [(qid,) + tuple(self.question_index(q, labels=labels, warm=False)) for qid, q in questions.items()]
        if order not in ("auto", "options", "state"):
            raise ValueError(f"unknown order {order!r}")
        if order == "state":
            shared = {qid for qid, _, _, _ in items}
        elif order == "auto":
            short = {qid for qid, _, idx, _ in items if len(idx._options) <= 2000
                     or len(idx.prefix) < len(idx.state_block(state))}
            shared = short if len(short) >= 2 else set()
        else:
            shared = set()
        pre = {}
        held = None
        if shared:
            first = next(idx for qid, _, idx, _ in items if qid in shared)
            held = first.segments(state, None, "shared-state")[0]
            self.client.hold(held)
        try:
            for qid, q, idx, instructions in items:
                if qid not in shared and getattr(idx, "prompt_tokens", 0) is None:
                    idx.warm()
            batch = [(qid, idx, ins) for qid, q, idx, ins in items
                     if qid in shared and idx.labels and not (full and q.type == "choice")]
            if len(batch) >= 2 and hasattr(self.client, "label_logprobs_tree"):
                reqs = [idx.label_request(state, ins, idx.keys, "shared-state") for _, idx, ins in batch]
                results = self.client.label_logprobs_tree(reqs)
                for (qid, idx, _), (_, lab), (lps, st) in zip(batch, reqs, results):
                    Index._account(st, usage)
                    pre[qid] = idx.label_answer(lps, lab)
            for qid, q, idx, instructions in items:
                if qid in pre:
                    a = pre[qid]
                else:
                    a = idx.choose(
                        state,
                        instructions,
                        usage,
                        full=full and q.type == "choice",
                        order="shared-state" if qid in shared else "options-first"
                    )
                if q.type == "choice":
                    if temperature != 1.0 and a.probabilities:
                        a = _retemper(a, temperature)
                    answers[qid] = a
                elif q.type == "score":
                    levels = idx.keys
                    probs = {lv: a.probabilities.get(lv, 0.0) for lv in levels}
                    answers[qid] = ScoreAnswer(
                        score=round(sum(int(k_) * p for k_, p in probs.items()), 4),
                        probabilities=probs,
                        confidence=a.confidence,
                        legend=dict(zip(levels, q.criteria))
                    )
                else:
                    answers[qid] = NoulAnswer(noul=round(a.probabilities.get("Yes", 0.0), 6))
        finally:
            if held is not None:
                self.client.release(held)
        return Response(
            model=f"kvindex/{self.model}",
            answers=answers,
            usage=usage,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2)
        )

    def train_questions(self, examples: Any, out: str | None = None, layout: str = "per-question",
                        **fit_kw) -> Any:
        """Train the embedding catalogue on typed questions. `examples`: (state, question, answer) triples; the
        question is a `Choice`, `Score` or `Noul` and the answer is anything `answer_target` accepts, soft
        distributions included (for instance the answers of `KV.teach`). Returns a head for
        `KV.decide(..., head=head)`; `out` saves it to a .npz file. `layout="per-question"` (default) embeds
        the state with every question; "state-once" embeds each state once and puts the question in the option
        texts: about six times faster at decision time, a few points less accurate (see the results)."""
        import numpy as np

        if layout not in ("state-once", "per-question"):
            raise ValueError(f"unknown layout {layout!r}")
        from .catalog import QUERY_TEMPLATE
        from .train import fit_head_grouped
        template = self.query_template or QUERY_TEMPLATE
        texts, row, candidates, targets = [], {}, [], []
        srow, stexts, sidx = {}, [], []
        for state, q, answer in examples:
            keys, opt_texts, instructions = catalog_texts(q, layout)
            for t in opt_texts:
                if t not in row:
                    row[t] = len(texts)
                    texts.append(t)
            candidates.append([row[t] for t in opt_texts])
            targets.append(answer_target(q, keys, answer))
            question = STATE_INSTRUCTION if layout == "state-once" else instructions or ""
            state_text = template.format(question=question, state=render_state(state))
            if state_text not in srow:
                srow[state_text] = len(stexts)
                stexts.append(state_text)
            sidx.append(srow[state_text])
        S = self.encoder(stexts)
        head = fit_head_grouped(S[np.asarray(sidx)], self.encoder(texts), candidates, targets, **fit_kw)
        head.report["layout"] = layout
        if out:
            head.save(out)
        return head

    def teach(self, cases: Any) -> list:
        """(state, question, answer) examples labelled by the LLM: for each (state, questions) pair, `decide` with
        the probability of every option. Use them with `train_questions` to train the catalogue without labels."""
        examples = []
        for state, questions in cases:
            r = self.decide(state, questions, full=True)
            for qid, q in questions.items():
                examples.append((state, q, r.answers[qid]))
        return examples

    def _decide_catalog(self, state: Any, questions: dict[str, Any], head: Any) -> Response:
        """Answer every question with the trained catalogue alone (no LLM call)."""
        import numpy as np

        from .catalog import QUERY_TEMPLATE
        t0 = time.perf_counter()
        template = self.query_template or QUERY_TEMPLATE
        scale = float(head.report.get("scale", 30.0))
        layout = head.report.get("layout", "per-question")
        items = [(qid, _coerce(q)) for qid, q in questions.items()]
        opts = [catalog_texts(q, layout) for _, q in items]
        if layout == "state-once":
            state_text = template.format(question=STATE_INSTRUCTION, state=render_state(state))
            z = head.state(self.encoder([state_text]))[0]
            zs = [z] * len(items)
        else:
            queries = [template.format(question=ins or "", state=render_state(state)) for _, _, ins in opts]
            zs = head.state(self.encoder(queries))
        cache = self.__dict__.setdefault("_option_vectors", {})
        missing = sorted({t for _, texts, _ in opts for t in texts if t not in cache})
        if missing:
            for t, v in zip(missing, self.encoder(missing)):
                cache[t] = v
        answers = {}
        for (qid, q), (keys, texts, _), z in zip(items, opts, zs):
            zo = head.options(np.stack([cache[t] for t in texts]))
            lp = {k: scale * float(s) for k, s in zip(keys, zo @ z)}
            probs = softmax(lp)
            conf = round(confidence(list(probs.values())), 4)
            if q.type == "choice":
                best = max(probs, key=probs.get)
                answers[qid] = ChoiceAnswer(choice=best, probabilities={k: round(p, 6) for k, p in probs.items()},
                                            confidence=conf)
            elif q.type == "score":
                answers[qid] = ScoreAnswer(score=round(sum(int(k) * p for k, p in probs.items()), 4),
                                           probabilities={k: round(p, 6) for k, p in probs.items()}, confidence=conf,
                                           legend=dict(zip(keys, q.criteria)))
            else:
                answers[qid] = NoulAnswer(noul=round(probs["Yes"], 6))
        return Response(model=f"kvindex/{self.model}/catalogue", answers=answers, usage=Usage(),
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    def question_index(self, q: Any, labels: bool = False, warm: bool = True) -> tuple[Any, Index, str]:
        """(question, index, instructions) used by `decide` for one typed question.

        Choice: the options (answer by name; labels=True: one-token labels, a probability for every option).
        Score: levels "0".."n" described by the criteria. Noul: "Yes"/"No" described by the criteria."""
        q = _coerce(q)
        if q.type == "choice":
            opts = q.options if q.options is not None else q.criteria
            idx = opts if isinstance(opts, Index) else self.index(opts, labels=labels, warm=warm)
            return q, idx, q.instructions
        if q.type == "score":
            levels = [str(i) for i in range(len(q.criteria))]
            idx = self.index({lv: c for lv, c in zip(levels, q.criteria)}, labels=levels, noun="level", warm=warm)
            return q, idx, q.instructions + " Answer with the level number."
        if q.type == "noul":
            crit = q.criteria or {}
            idx = self.index({"Yes": crit.get("true") or "the answer to the question is yes / true",
                              "No": crit.get("false") or "the answer to the question is no / false"},
                             labels=["Yes", "No"], noun="answer", warm=warm)
            return q, idx, q.instructions
        raise ValueError(f"unknown question type {q.type!r}")


STATE_INSTRUCTION = "Answer the questions about this state"

def catalog_texts(q: Any, layout: str) -> tuple[list[str], list[str], str]:
    """Option keys, option texts and instructions of a typed question for the trained catalogue in `layout`."""
    keys, texts, instructions = question_options(q)
    if layout == "state-once" and instructions:
        texts = [instructions + "\n" + t for t in texts]
    return keys, texts, instructions


def question_options(q: Any) -> tuple[list[str], list[str], str]:
    """Option keys, option texts for the embedding catalogue, and instructions of one typed question."""
    q = _coerce(q)
    if q.type == "choice":
        opts = q.options if q.options is not None else q.criteria
        if hasattr(opts, 'keys') and isinstance(opts.keys, list) and hasattr(opts, 'texts'):
            keys = opts.keys
            texts = opts.texts
        else:
            keys, _, texts = parse_options(opts)
        return keys, texts, q.instructions
    elif q.type == "score":
        keys = [str(i) for i in range(len(q.criteria))]
        texts = [str(c) for c in q.criteria]
        return keys, texts, q.instructions
    elif q.type == "noul":
        crit = q.criteria or {}
        texts = ["Yes: " + (crit.get("true") or "the answer to the question is yes / true"),
                 "No: " + (crit.get("false") or "the answer to the question is no / false")]
        return ["Yes", "No"], texts, q.instructions
    else:
        raise ValueError(f"unknown question type {q.type!r}")


def answer_target(q: Any, keys: list[str], answer: Any) -> list[float]:
    """Target distribution over `keys` for one answer: an option name, a level, a bool or a probability, a
    {key: probability} dict, or an answer object returned by `KV.decide`."""
    q = _coerce(q)
    if isinstance(answer, ChoiceAnswer):
        if len(answer.probabilities) > 1:
            probs = answer.probabilities
        else:
            probs = {answer.choice: 1.0}
    elif isinstance(answer, ScoreAnswer):
        probs = {str(k): v for k, v in answer.probabilities.items()}
    elif isinstance(answer, NoulAnswer):
        probs = {"Yes": answer.noul, "No": 1 - answer.noul}
    elif isinstance(answer, dict):
        if q.type == "noul":
            probs = {}
            for k, v in answer.items():
                k_stripped_lower = k.strip().lower()
                if k_stripped_lower in ("true", "yes"):
                    probs["Yes"] = v
                elif k_stripped_lower in ("false", "no"):
                    probs["No"] = v
                else:
                    probs[k] = v
        elif q.type == "score":
            probs = {str(k): v for k, v in answer.items()}
        else:
            probs = answer.copy()
    elif isinstance(answer, bool):
        if q.type != "noul":
            raise ValueError("bool answers are only valid for noul questions")
        probs = {"Yes": 1.0, "No": 0.0} if answer else {"Yes": 0.0, "No": 1.0}
    elif isinstance(answer, (int, float)):
        if q.type == "noul":
            f = float(answer)
            probs = {"Yes": f, "No": 1 - f}
        elif q.type == "score":
            k = str(int(round(answer)))
            probs = {k: 1.0}
        else:
            raise ValueError("numeric answers are only valid for noul or score questions")
    elif isinstance(answer, str):
        if q.type == "noul":
            answer_stripped_lower = answer.strip().lower()
            if answer_stripped_lower in ("true", "yes"):
                probs = {"Yes": 1.0, "No": 0.0}
            elif answer_stripped_lower in ("false", "no"):
                probs = {"Yes": 0.0, "No": 1.0}
            else:
                probs = {answer: 1.0}
        else:
            probs = {answer: 1.0}
    else:
        raise ValueError(f"unknown answer type {type(answer)}")

    total = sum(probs.get(k, 0.0) for k in keys)
    if total == 0.0:
        raise ValueError(f"answer {answer!r} gives no probability to any option")
    return [float(probs.get(k, 0.0)) / total for k in keys]


def _coerce(q: Any):
    """Accept kvindex dataclasses, OpenJev pydantic models or plain dicts."""
    if isinstance(q, (Choice, Score, Noul)):
        return q
    d = q if isinstance(q, dict) else (q.model_dump() if hasattr(q, "model_dump") else vars(q))
    t = d.get("type")
    if t == "choice":
        opts = d.get("options")
        if opts is None:
            opts = d.get("criteria")
        return Choice(instructions=d.get("instructions", ""), options=opts)
    if t == "score":
        crit = d.get("criteria")
        if crit is None and d.get("legend"):
            crit = [d["legend"][k] for k in sorted(d["legend"], key=int)]
        return Score(instructions=d.get("instructions", ""), criteria=list(crit))
    if t == "noul":
        crit = d.get("criteria")
        return Noul(instructions=d.get("instructions", ""), criteria=crit if isinstance(crit, dict) else None)
    raise ValueError(f"unknown question {q!r}")


def _retemper(a: ChoiceAnswer, temperature: float) -> ChoiceAnswer:
    lp = {k: math.log(p) / temperature for k, p in a.probabilities.items() if p > 0}
    probs = softmax(lp)
    full = {k: round(probs.get(k, 0.0), 6) for k in a.probabilities}
    return ChoiceAnswer(choice=a.choice, probabilities=full, confidence=round(confidence(list(full.values())), 4))


def iter_answers(index: Index, states: Iterable[Any], instructions: str | None = None):
    """Answer many states against one index (consecutive calls keep the options cached)."""
    for s in states:
        yield index.choose(s, instructions)
