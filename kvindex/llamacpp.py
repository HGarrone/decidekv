"""
In-process llama.cpp backend.

* Loads the GGUF file that Ollama already downloaded (`KV("qwen3:8b", backend="llamacpp")`) or any GGUF path.
* One llama.cpp context; its KV cache holds the tokens of the last prompt.
* Only new tokens are evaluated: every call reuses the longest common token prefix already in the cache and cuts
  the rest in O(1). All logits are available (Ollama returns at most 20 log-probabilities).
* Answers by name are decoded through a token trie of the options: the output is always one of the options.

Needs llama-cpp-python (`pip install "kvindex[llamacpp]"`; on Windows also `python -m kvindex.get_llamacpp`).
Pure-attention models only: recurrent layers (Qwen3.5/3.8, Mamba, ...) cannot be cut back to a prefix.
"""
from __future__ import annotations

import ctypes
import glob
import json
import math
import os
import sys

KV_TYPES = {"f16": 1, "q8_0": 8}          # ggml type ids


# ---------------------------------------------------------------- locating Ollama models and the llama.cpp library
def ollama_models_dir() -> str:
    return os.environ.get("OLLAMA_MODELS") or os.path.join(os.path.expanduser("~"), ".ollama", "models")


def ollama_gguf(name: str) -> str:
    """Path of the GGUF blob for an Ollama model name such as 'qwen3:8b' or 'hf.co/user/repo:Q4_K_M'."""
    if os.path.isfile(name):
        return name
    model, _, tag = name.partition(":")
    tag = tag or "latest"
    parts = model.split("/")
    if len(parts) == 1:
        parts = ["registry.ollama.ai", "library", parts[0]]
    elif len(parts) == 2:
        parts = ["registry.ollama.ai"] + parts
    man = os.path.join(ollama_models_dir(), "manifests", *parts, tag)
    if not os.path.isfile(man):
        raise FileNotFoundError(f"Ollama manifest not found for {name!r} ({man}). Run `ollama pull {name}` "
                                "or pass a .gguf path.")
    layers = json.load(open(man))["layers"]
    digest = next(layer["digest"] for layer in layers if layer["mediaType"].endswith(".model"))
    return os.path.join(ollama_models_dir(), "blobs", digest.replace(":", "-"))


def _prepare_dlls():
    """Use an official llama.cpp build when one is installed (KVINDEX_LLAMA_LIB, ~/.kvindex, or ./vendor): needed
    when the llama-cpp-python wheel was compiled for CPU instructions this machine lacks. Otherwise, on Windows,
    make the CUDA runtime shipped with Ollama visible to the wheel."""
    lib = os.environ.get("KVINDEX_LLAMA_LIB") or os.environ.get("LLAMA_CPP_LIB_PATH")
    if not lib:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        found = sorted(glob.glob(os.path.join(os.path.expanduser("~"), ".kvindex", "llama.cpp-*", "bin", "llama.dll"))
                       + glob.glob(os.path.join(here, "vendor", "llama.cpp-*", "bin", "llama.dll")))
        lib = os.path.dirname(found[-1]) if found else None
    if lib:
        os.environ["LLAMA_CPP_LIB_PATH"] = lib
        os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
        ggml = os.path.join(lib, "ggml.dll" if sys.platform == "win32" else "libggml.so")
        if os.path.exists(ggml):          # official builds load the CPU/CUDA backends as separate modules
            ctypes.CDLL(ggml).ggml_backend_load_all_from_path(lib.encode())
        return
    if sys.platform != "win32":
        return
    import importlib.util
    spec = importlib.util.find_spec("llama_cpp")
    extra = [os.path.join(os.path.dirname(spec.origin), "lib")] if spec and spec.origin else []
    base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "lib", "ollama")
    extra += sorted(glob.glob(os.path.join(base, "cuda_v12")))
    os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])


def _llama_cpp():
    _prepare_dlls()
    try:
        import llama_cpp
    except ImportError as e:  # pragma: no cover
        raise ImportError('backend="llamacpp" needs llama-cpp-python: pip install "kvindex[llamacpp]"') from e
    return llama_cpp


# ---------------------------------------------------------------- generation backend
class _Trie:
    __slots__ = ("kids", "end", "opts")

    def __init__(self):
        self.kids, self.end, self.opts = {}, None, set()      # opts: every option reachable below this node


class LlamaCpp:
    """Backend for `KV(model, backend="llamacpp")`."""

    def __init__(self, model: str, n_ctx: int = 16384, n_batch: int = 2048, n_gpu_layers: int = -1,
                 kv_type: str = "f16", verbose: bool = False):
        L = _llama_cpp()
        self.L = L
        self.model, self.path = model, ollama_gguf(model)
        # weights, tokenizer and metadata; the high-level object's own context is tiny and never used
        self.llm = L.Llama(self.path, n_ctx=64, n_batch=64, n_gpu_layers=n_gpu_layers, verbose=verbose)
        p = L.llama_context_default_params()
        p.n_ctx, p.n_batch, p.n_ubatch = n_ctx, n_batch, min(512, n_batch)
        p.n_seq_max, p.kv_unified = 2, True                     # the measured configuration; only sequence 0 is used
        p.flash_attn_type = L.LLAMA_FLASH_ATTN_TYPE_ENABLED
        p.type_k = p.type_v = KV_TYPES[kv_type]
        p.offload_kqv, p.no_perf = True, True
        self.ctx = L.llama_init_from_model(self.llm.model, p)
        if not self.ctx:
            raise RuntimeError(f"llama.cpp could not create a context with n_ctx={n_ctx} (out of memory?)")
        self.mem = L.llama_get_memory(self.ctx)
        self.batch = L.llama_batch_init(n_batch, 0, 1)
        self.n_ctx, self.n_batch, self.n_vocab, self.kv_type = n_ctx, n_batch, self.llm.n_vocab(), kv_type
        self.cur: list[int] = []                                # tokens of sequence 0, at positions 0..len-1
        self._tok_cache: dict[str, list[int]] = {}
        self._tries: dict[tuple, _Trie] = {}
        self._stops = None
        self._needs_bos = None
        self._wrap = self._chat_wrap()

    def close(self):
        if getattr(self, "ctx", None):
            self.L.llama_batch_free(self.batch)
            self.L.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "llm", None) is not None:
            self.llm.close()
            self.llm = None

    # ---------------------------------------------------------- low level: decode and KV-cache primitives
    def _decode(self, tokens: list[int], pos0: int, seq: int = 0):
        """Evaluate `tokens` at positions pos0.. in sequence `seq`; logits are kept for the last token only."""
        b = self.batch
        for i in range(0, len(tokens), self.n_batch):
            chunk = tokens[i:i + self.n_batch]
            b.n_tokens = len(chunk)
            for j, t in enumerate(chunk):
                b.token[j], b.pos[j], b.n_seq_id[j], b.logits[j] = t, pos0 + i + j, 1, 0
                b.seq_id[j][0] = seq
            if i + len(chunk) == len(tokens):
                b.logits[len(chunk) - 1] = 1
            rc = self.L.llama_decode(self.ctx, b)
            if rc != 0:
                raise RuntimeError(f"llama_decode failed with code {rc}")

    def seq_rm(self, seq: int, p0: int = 0, p1: int = -1):
        self.L.llama_memory_seq_rm(self.mem, seq, p0, p1)

    def clear(self):
        self.L.llama_memory_clear(self.mem, True)
        self.cur = []

    # ---------------------------------------------------------- tokens
    def _chat_wrap(self) -> tuple[str, str]:
        """Text before and after the user message in the model's own chat template (thinking disabled)."""
        tpl = self.llm.metadata.get("tokenizer.chat_template")
        if not tpl:
            return "", ""
        import jinja2
        marker = "\x00KVINDEX\x00"
        out = jinja2.Environment().from_string(tpl).render(
            messages=[{"role": "user", "content": marker}], add_generation_prompt=True, enable_thinking=False,
            bos_token=self.llm.detokenize([self.llm.token_bos()]).decode(errors="ignore"), eos_token="")
        before, after = out.split(marker)
        return before, after

    def tokenize(self, text: str, special: bool = False) -> list[int]:
        return self.llm.tokenize(text.encode(), add_bos=False, special=special)

    def _tokens(self, text: str) -> list[int]:
        t = self._tok_cache.get(text)
        if t is None:
            t = self.tokenize(text, special=True)
            if len(self._tok_cache) > 4096:
                self._tok_cache.clear()
            self._tok_cache[text] = t
        return t

    def bos(self) -> list[int]:
        if self._needs_bos is None:
            probe = self.llm.tokenize(b"x", add_bos=True, special=False)
            self._needs_bos = len(probe) > 1 and probe[0] == self.llm.token_bos()
        return [self.llm.token_bos()] if self._needs_bos else []

    def prompt_tokens(self, segments) -> list[int]:
        """Tokens of a prompt made of segments, wrapped in the chat template; segments are tokenised separately
        so that a shared first segment always produces the same token prefix."""
        segs = list(segments)
        segs[0] = self._wrap[0] + segs[0]
        segs[-1] = segs[-1] + self._wrap[1]
        toks = []
        for s in segs:
            toks += self._tokens(s)
        bos = self.bos()
        if bos and (not toks or toks[0] != bos[0]):
            toks = bos + toks
        return toks

    def eval_tokens(self, toks: list[int]) -> dict:
        """Reuse the longest common prefix with sequence 0, evaluate only the rest."""
        if len(toks) + 64 > self.n_ctx:
            raise ValueError(f"prompt of {len(toks)} tokens does not fit n_ctx={self.n_ctx}")
        common, lim = 0, min(len(self.cur), len(toks) - 1)     # always evaluate >= 1 token for fresh logits
        while common < lim and self.cur[common] == toks[common]:
            common += 1
        self.seq_rm(0, common, -1)
        self._decode(toks[common:], common, 0)
        self.cur = list(toks)
        return {"input_tokens": len(toks) - common, "cached_tokens": common, "output_tokens": 0}

    def _append(self, tok: int):
        self._decode([tok], len(self.cur), 0)
        self.cur.append(tok)

    def logprobs_last(self):
        import numpy as np
        ptr = self.L.llama_get_logits_ith(self.ctx, -1)
        x = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).astype(np.float64)
        m = x.max()
        return x - (m + math.log(np.exp(x - m).sum()))

    # ---------------------------------------------------------- backend contract
    def architecture(self) -> str:
        return str(self.llm.metadata.get("general.architecture", ""))

    def prefill(self, segments, num_ctx=None) -> dict:
        return self.eval_tokens(self.prompt_tokens(segments))

    def label_logprobs(self, segments, labels, num_ctx=None):
        """log P of each one-token label; the label with and without a leading space are summed."""
        stats = self.eval_tokens(self.prompt_tokens(segments))
        lp = self.logprobs_last()
        out = {}
        for lab in labels:
            ids = {self.tokenize(v)[0] for v in (lab, " " + lab)}
            out[lab] = float(math.log(sum(math.exp(lp[i]) for i in ids)))
        stats["output_tokens"] = 1
        return out, stats

    def first_token_logprobs(self, segments, labels):
        """OpenJev's readout: logit of the first token of each label (tokenised alone), as log-probabilities."""
        stats = self.eval_tokens(self.prompt_tokens(segments))
        lp = self.logprobs_last()
        out = {lab: float(lp[self.tokenize(lab)[0]]) for lab in labels}
        stats["output_tokens"] = 1
        return out, stats

    def pick_option(self, segments, options, num_ctx=None, enum_max=None, allowed=None):
        """Greedy decoding constrained to the trie of `options` (of `allowed` when given)."""
        stats = self.eval_tokens(self.prompt_tokens(segments))
        text, total, steps = self.decode_option(options, allowed)
        stats["output_tokens"] = steps
        return text, total, stats

    def decode_option(self, options, allowed=None):
        """Continue sequence 0 under the option-trie constraint; -> (option, log-prob, steps)."""
        trie = self._trie(options)
        ok = set(allowed) if allowed is not None else None
        node, total, steps = trie, 0.0, 0
        stop_ids = self._stop_ids()
        while True:
            lp = self.logprobs_last()
            best_tok, best_lp = None, -1e30
            for t, child in node.kids.items():
                if ok is not None and child.opts.isdisjoint(ok):
                    continue
                if lp[t] > best_lp:
                    best_tok, best_lp = t, float(lp[t])
            if node.end is not None and (ok is None or node.end in ok):
                end_lp = max(float(lp[s]) for s in stop_ids)
                if best_tok is None or end_lp >= best_lp:
                    break
            if best_tok is None:
                break
            total += best_lp
            node = node.kids[best_tok]
            steps += 1
            self._append(best_tok)
        return node.end, total, steps

    def _stop_ids(self) -> set[int]:
        """Every end-of-generation token of the model (eos, <|eot_id|>, <|im_end|>, ...) plus newline."""
        if self._stops is None:
            vocab = self.L.llama_model_get_vocab(self.llm.model)
            stops = {t for t in range(self.n_vocab) if self.L.llama_vocab_is_eog(vocab, t)}
            stops.add(self.llm.token_eos())
            stops.update(self.tokenize("\n")[-1:])
            self._stops = stops
        return self._stops

    def _trie(self, options) -> _Trie:
        key = tuple(options)
        t = self._tries.get(key)
        if t is None:
            t = _Trie()
            for o in options:
                cap = o[:1].upper() + o[1:]       # surface variants the model may start with; all map to `o`
                for v in dict.fromkeys((o, cap, " " + o, " " + cap)):
                    node = t
                    node.opts.add(o)
                    for tok in self.tokenize(v):
                        node = node.kids.setdefault(tok, _Trie())
                        node.opts.add(o)
                    if node.end is None:
                        node.end = o
            if len(self._tries) > 512:
                self._tries.clear()
            self._tries[key] = t
        return t


# ---------------------------------------------------------------- embedding encoder
class LlamaCppEncoder:
    """Sentence embeddings from a GGUF embedding model (e.g. Ollama's qwen3-embedding:4b), L2-normalised."""

    def __init__(self, model: str, n_ctx: int = 2048, n_gpu_layers: int = -1, pooling: str | None = None,
                 verbose: bool = False):
        L = _llama_cpp()
        kw = {}
        if pooling:
            kw["pooling_type"] = {"last": L.LLAMA_POOLING_TYPE_LAST, "mean": L.LLAMA_POOLING_TYPE_MEAN}[pooling]
        self.llm = L.Llama(ollama_gguf(model), embedding=True, n_ctx=n_ctx, n_batch=n_ctx, n_ubatch=n_ctx,
                           n_gpu_layers=n_gpu_layers, verbose=verbose, **kw)

    def __call__(self, texts, batch: int = 64):
        import numpy as np
        out = []
        for i in range(0, len(texts), batch):
            out += self.llm.embed(list(texts[i:i + batch]))
        e = np.asarray(out, dtype=np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)

    def close(self):
        self.llm.close()
