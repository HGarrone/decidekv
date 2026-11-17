"""
In-process llama.cpp backend.

* Loads the GGUF file that Ollama already downloaded (`KV("qwen3:8b", backend="llamacpp")`) or any GGUF path.
* One llama.cpp context. A prefix that repeats (the option list of an index, the state of a multi-question decision)
  is evaluated once in a sequence of its own and copied into the working sequence at every call, so only the new
  tokens are computed. This works on every architecture, including models with recurrent layers; several prefixes
  stay resident at the same time (`resident=`), least recently used evicted first.
* Where the model allows removing the tail of a sequence, other prompts also reuse the longest common token prefix
  with the previous one. All logits are available (Ollama returns at most 20 log-probabilities).
* Answers by name are decoded through a token trie of the options: the output is always one of the options.

Needs llama-cpp-python (`pip install "kvindex[llamacpp]"`; on Windows also `python -m kvindex.get_llamacpp`).
"""
from __future__ import annotations

import ctypes
import glob
import json
import math
import os
import sys

from .ollama import HYBRID_ARCHS

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
        v1 = ["registry.ollama.ai", "library", parts[0]]
        v2 = ["ollama.com", "library", parts[0]]
    elif len(parts) == 2:
        v1 = ["registry.ollama.ai"] + parts
        v2 = ["ollama.com"] + parts
    else:
        v1 = parts
        v2 = parts
    root = ollama_models_dir()
    candidates = [os.path.join(root, "manifests", *v1, tag), os.path.join(root, "manifests-v2", *v2, tag)]
    man = next((p for p in candidates if os.path.isfile(p)), None)
    if man is None:
        man = _find_ci(root, ["manifests-v2", *v2, tag])
    if man is None:
        man = _find_ci(root, ["manifests", *v1, tag])
    if man is None:
        raise FileNotFoundError(f"Ollama manifest not found for {name!r} (looked in {candidates[0]} and "
                                f"{candidates[1]}). Run `ollama pull {name}` or pass a .gguf path.")
    with open(man, encoding="utf8") as f:
        layers = json.load(f)["layers"]
    model_layer = next((layer for layer in layers if layer["mediaType"] == "application/vnd.ollama.image.model"),
                       None)
    if model_layer is None:
        model_layer = next((layer for layer in layers if layer["mediaType"].endswith(".model")), None)
    digest = model_layer["digest"]
    return os.path.join(ollama_models_dir(), "blobs", digest.replace(":", "-"))


def _find_ci(root: str, parts: list[str]) -> str | None:
    """Path under `root` made of `parts`, matching every folder and file name without regard to case, or None."""
    path = root
    for part in parts:
        if not os.path.isdir(path):
            return None
        names = os.listdir(path)
        hit = next((n for n in names if n.lower() == part.lower()), None)
        if hit is None:
            return None
        path = os.path.join(path, hit)
    return path if os.path.isfile(path) else None


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
                 kv_type: str = "f16", verbose: bool = False, resident: int = 7, kv_on_gpu: bool = True,
                 scratch: int | None = None):
        L = _llama_cpp()
        self.L = L
        self.model, self.path = model, ollama_gguf(model)
        # weights, tokenizer and metadata; the high-level object's own context is tiny and never used
        self.llm = L.Llama(self.path, n_ctx=64, n_batch=64, n_gpu_layers=n_gpu_layers, verbose=verbose)
        hybrid = str(self.llm.metadata.get("general.architecture", "")) in HYBRID_ARCHS
        self.scratch = (0 if hybrid else 32) if scratch is None else scratch   # sequences for batched option scoring
        p = L.llama_context_default_params()
        p.n_ctx, p.n_batch, p.n_ubatch = n_ctx, n_batch, min(512, n_batch)
        p.n_seq_max, p.kv_unified = resident + 1 + self.scratch, True   # 0 works, then resident prefixes, then scratch
        # more than 8 sequences in total make models with recurrent layers up to 10x slower (llama.cpp b11312)
        p.flash_attn_type = L.LLAMA_FLASH_ATTN_TYPE_ENABLED
        p.type_k = p.type_v = KV_TYPES[kv_type]
        p.offload_kqv, p.no_perf = kv_on_gpu, True
        self.ctx = L.llama_init_from_model(self.llm.model, p)
        if not self.ctx:
            raise RuntimeError(f"llama.cpp could not create a context with n_ctx={n_ctx} (out of memory?)")
        self.mem = L.llama_get_memory(self.ctx)
        self.batch = L.llama_batch_init(n_batch, 0, 1)
        self.n_ctx, self.n_batch, self.n_vocab, self.kv_type = n_ctx, n_batch, self.llm.n_vocab(), kv_type
        self.cur: list[int] = []                                # tokens of sequence 0, at positions 0..len-1
        self.resident = resident
        self.bases: dict[int, list[int]] = {}                   # resident prefixes: sequence -> tokens, oldest first
        self._registered: dict[str, None] = {}                  # first-segment texts declared resident by prefill()
        self.can_cut = bool(L.llama_memory_can_shift(self.mem))  # False for models with recurrent layers
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

    def seq_cp(self, src: int, dst: int):
        """Copy a whole sequence into another one (cache cells are shared, a recurrent state is copied)."""
        self.L.llama_memory_seq_cp(self.mem, src, dst, -1, -1)

    def clear(self):
        self.L.llama_memory_clear(self.mem, True)
        self.cur = []
        self.bases = {}

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

    def _split(self, segments) -> tuple[list[int], int]:
        """Tokens of a prompt made of segments (wrapped in the chat template) and the number of tokens of its first
        segment. Segments are tokenised separately, so a shared first segment always gives the same token prefix."""
        segs = list(segments)
        segs[0] = self._wrap[0] + segs[0]
        segs[-1] = segs[-1] + self._wrap[1]
        parts = [self._tokens(s) for s in segs]
        toks = [tok for part in parts for tok in part]
        n = len(parts[0])
        bos = self.bos()
        if bos and (not toks or toks[0] != bos[0]):
            toks = bos + toks
            n += len(bos)
        return toks, n

    def prompt_tokens(self, segments) -> list[int]:
        """Tokens of a prompt made of segments, wrapped in the chat template."""
        return self._split(segments)[0]

    def _room(self, need: int, keep: int | None = None):
        """Evict the least recently used resident prefixes until `need` more tokens fit in the context."""
        while sum(len(t) for t in self.bases.values()) + need + 64 > self.n_ctx:
            victim = next((s for s in self.bases if s != keep), None)
            if victim is None:
                raise ValueError(f"prompt of {need} tokens does not fit n_ctx={self.n_ctx}")
            self.seq_rm(victim, -1, -1)
            del self.bases[victim]

    def _base(self, prefix: list[int]) -> tuple[int, int]:
        """Sequence that holds `prefix` as a resident prefix, and the number of tokens evaluated to create it (0 when
        it was already there)."""
        for seq, toks in self.bases.items():
            if toks == prefix:
                self.bases[seq] = self.bases.pop(seq)
                return seq, 0
        self._room(len(prefix))
        if len(self.bases) >= self.resident:
            victim = next(iter(self.bases))
            self.seq_rm(victim, -1, -1)
            del self.bases[victim]
        seq = next(s for s in range(1, self.resident + 1) if s not in self.bases)
        self.seq_rm(seq, -1, -1)
        self._decode(prefix, 0, seq)
        self.bases[seq] = list(prefix)
        return seq, len(prefix)

    def eval_tokens(self, toks: list[int], base_len: int = 0) -> dict:
        """Evaluate a prompt in sequence 0. With `base_len`, the first `base_len` tokens are a resident prefix: they
        are evaluated once in a sequence of their own and copied into sequence 0 at every call (every architecture).
        Without it, where the model allows, the longest common prefix with the previous prompt is reused."""
        if len(toks) + 64 > self.n_ctx:
            raise ValueError(f"prompt of {len(toks)} tokens does not fit n_ctx={self.n_ctx}")
        if 0 < base_len < len(toks):
            self.seq_rm(0, -1, -1)
            self.cur = []
            seq, made = self._base(toks[:base_len])
            self._room(len(toks) - base_len, keep=seq)
            self.seq_cp(seq, 0)
            self._decode(toks[base_len:], base_len, 0)
            self.cur = list(toks)
            return {"input_tokens": len(toks) - base_len + made, "cached_tokens": base_len - made, "output_tokens": 0}
        else:
            common = 0
            if self.can_cut:
                lim = min(len(self.cur), len(toks) - 1)
                while common < lim and self.cur[common] == toks[common]:
                    common += 1
                self.seq_rm(0, common, -1)
            else:
                self.seq_rm(0, -1, -1)
            self.cur = self.cur[:common]
            self._room(len(toks))
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

    def _eval(self, segments) -> dict:
        """Evaluate a prompt; its first segment is used as a resident prefix when it was declared with `prefill`."""
        toks, n = self._split(segments)
        return self.eval_tokens(toks, n if segments[0] in self._registered else 0)

    def prefill(self, segments, num_ctx=None) -> dict:
        """Declare the first segment a resident prefix and evaluate it (the offline step of an index)."""
        key = segments[0]
        self._registered.pop(key, None)
        self._registered[key] = None
        while len(self._registered) > 64:
            self._registered.pop(next(iter(self._registered)))
        return self._eval(segments)

    def label_logprobs(self, segments, labels, num_ctx=None):
        """log P of each one-token label; the label with and without a leading space are summed."""
        stats = self._eval(segments)
        lp = self.logprobs_last()
        out = {}
        for lab in labels:
            ids = {self.tokenize(v)[0] for v in (lab, " " + lab)}
            out[lab] = float(math.log(sum(math.exp(lp[i]) for i in ids)))
        stats["output_tokens"] = 1
        return out, stats

    def first_token_logprobs(self, segments, labels):
        """OpenJev's readout: logit of the first token of each label (tokenised alone), as log-probabilities."""
        stats = self._eval(segments)
        lp = self.logprobs_last()
        out = {lab: float(lp[self.tokenize(lab)[0]]) for lab in labels}
        stats["output_tokens"] = 1
        return out, stats

    def pick_option(self, segments, options, num_ctx=None, enum_max=None, allowed=None):
        """Greedy decoding constrained to the trie of `options` (of `allowed` when given)."""
        stats = self._eval(segments)
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

    def _paths(self, root, ok):
        """Every token path from the root of the option trie to a node that ends an option of `ok`, with that
        option: a list of (tokens, option)."""
        result = []
        stack = [(root, [])]
        while stack:
            node, path = stack.pop()
            if node.opts.isdisjoint(ok):
                continue
            if node.end is not None and node.end in ok:
                result.append((path, node.end))
            if node.kids:
                for tok, child in node.kids.items():
                    stack.append((child, path + [tok]))
        return result

    def _score_batched(self, paths, L, lp0, seqs):
        """log P of each path after the prompt in sequence 0 (length L): many paths per llama_decode, each in one of
        the free sequences `seqs`. `lp0` are the log-probabilities after the prompt. Returns one float per path."""
        import numpy as np
        out = [0.0] * len(paths)
        i = 0
        while i < len(paths):
            j = i
            used = 0
            while j < len(paths) and j - i < len(seqs) and used + len(paths[j][0]) <= self.n_batch:
                used += len(paths[j][0])
                j += 1
            if j == i:
                raise ValueError("option name longer than n_batch tokens")
            b = self.batch
            n = 0
            rows = []
            k_out = 0
            for c, (toks, _) in enumerate(paths[i:j]):
                seq = seqs[c]
                self.seq_rm(seq, -1, -1)
                self.seq_cp(0, seq)
                r = []
                for t_i, tok in enumerate(toks):
                    b.token[n] = tok
                    b.pos[n] = L + t_i
                    b.n_seq_id[n] = 1
                    b.seq_id[n][0] = seq
                    keep = t_i < len(toks) - 1
                    b.logits[n] = 1 if keep else 0
                    if keep:
                        r.append(k_out)
                        k_out += 1
                    n += 1
                rows.append(r)
            b.n_tokens = n
            if k_out > 0:
                rc = self.L.llama_decode(self.ctx, b)
                if rc != 0:
                    raise RuntimeError(f"llama_decode failed with code {rc}")
                ptr = self.L.llama_get_logits(self.ctx)
                x = np.ctypeslib.as_array(ptr, shape=(k_out, self.n_vocab))
                m = x.max(axis=1)
                e = x - m[:, None]
                np.exp(e, out=e)
                lse = m + np.log(e.sum(axis=1))
            else:
                x = None
                lse = None
            for c, (toks, _) in enumerate(paths[i:j]):
                s = float(lp0[toks[0]])
                if x is not None:
                    for t_i in range(1, len(toks)):
                        row = rows[c][t_i - 1]
                        s += float(x[row, toks[t_i]] - lse[row])
                out[i + c] = s
            for c in range(j - i):
                self.seq_rm(seqs[c], -1, -1)
            i = j
        return out

    def score_options(self, segments, options, allowed=None, floor: float = 1e-6):
        """log P of every option (sum over its tokens, summed over its surface variants), normalised over the options
        of `allowed` (all of them when None), and the stats of the prompt evaluation. Exact on every architecture.
        Many option paths are evaluated in one call, each in a free sequence (the reserved `scratch` ones and the
        resident slots not in use); with no free sequence the trie is walked one node at a time. A path whose
        probability falls below `floor` is not followed further (after its first token in the batched mode, at any
        depth in the walk): its value is that probability, an upper bound with an error below `floor`."""
        stats = self._eval(segments)
        prompt = list(self.cur)
        L = len(prompt)
        ok = set(allowed) if allowed is not None else set(options)
        root = self._trie(options)
        mass = {}
        base = None if self.can_cut else self._base(prompt)[0]
        steps = 0

        def logits_after(path):
            nonlocal steps
            if self.can_cut:
                if path:
                    self.seq_rm(0, L + len(path) - 1, -1)
                    self._decode(path[-1:], L + len(path) - 1, 0)
            else:
                self.seq_rm(0, -1, -1)
                self.seq_cp(base, 0)
                if path:
                    self._decode(path, L, 0)
            steps += 1
            return self.logprobs_last()

        def visit(node, path, acc, lp):
            for tok, child in node.kids.items():
                if child.opts.isdisjoint(ok):
                    continue
                val = acc + float(lp[tok])
                if val < math.log(floor):
                    if child.end is None or child.end not in ok:
                        for name in child.opts & ok:
                            mass[name] = val if name not in mass else max(mass[name], val)
                        continue
                if child.end is not None and child.end in ok:
                    if child.end not in mass:
                        mass[child.end] = val
                    else:
                        m = max(mass[child.end], val)
                        mass[child.end] = math.log(math.exp(mass[child.end] - m) + math.exp(val - m)) + m
                if child.kids and not all(c.opts.isdisjoint(ok) for c in child.kids.values()):
                    visit(child, path + [tok], val, logits_after(path + [tok]))

        lp = self.logprobs_last()
        free = [s for s in range(1, self.resident + 1 + self.scratch) if s not in self.bases]
        if free:
            paths = self._paths(root, ok)
            cut = math.log(floor)
            deep = [p for p in paths if float(lp[p[0][0]]) >= cut]
            vals = dict(zip(map(id, deep), self._score_batched(deep, L, lp, free)))
            for p in paths:
                toks, name = p
                val = vals[id(p)] if id(p) in vals else float(lp[toks[0]])
                if name not in mass:
                    mass[name] = val
                else:
                    m = max(mass[name], val)
                    mass[name] = math.log(math.exp(mass[name] - m) + math.exp(val - m)) + m
            steps = -(-len(deep) // len(free))
        else:
            visit(root, [], 0.0, lp)

        if self.can_cut:
            self.seq_rm(0, L, -1)
        else:
            self.seq_rm(0, -1, -1)
            self.seq_cp(base, 0)
        self.cur = prompt

        names = [o for o in options if o in ok]
        z = max(mass.get(o, -1e30) for o in names)
        logz = z + math.log(sum(math.exp(mass.get(o, -1e30) - z) for o in names))
        out = {o: mass.get(o, -1e30) - logz for o in names}
        stats["output_tokens"] = steps
        return out, stats

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

    def hold(self, prefix: str) -> None:
        """Declare `prefix` a resident prefix without evaluating anything: the next prompt that starts with it
        evaluates it once, and the following ones copy it (for instance one state shared by several questions)."""
        key = prefix
        self._registered.pop(key, None)
        self._registered[key] = None
        while len(self._registered) > 64:
            self._registered.pop(next(iter(self._registered)))

    def release(self, prefix: str) -> None:
        """Forget a resident prefix declared with `hold` and free its cache cells."""
        self._registered.pop(prefix, None)
        toks, n = self._split([prefix, ""])
        head = toks[:n]
        for seq in [s for s, t in self.bases.items() if t == head]:
            self.seq_rm(seq, -1, -1)
            del self.bases[seq]

    def label_logprobs_tree(self, items):
        """Single-token label log-probabilities for several prompts that share their first segment (one state) and
        may share their second one (one option list): the state is evaluated once, each distinct second segment
        once, and the last segments of all the prompts in parallel sequences. `items` is a list of
        (segments, labels); returns one (log-probabilities, stats) per item, like `label_logprobs`."""
        import numpy as np
        if not items:
            return []
        s0 = items[0][0][0]
        if any(item[0][0] != s0 for item in items):
            raise ValueError("label_logprobs_tree needs prompts with the same first segment")
        p0 = self._tokens(self._wrap[0] + s0)
        if self.bos() and (not p0 or p0[0] != self.bos()[0]):
            p0 = self.bos() + p0
        seq_b, made = self._base(p0)
        n0 = len(p0)
        free = [s for s in range(1, self.resident + self.scratch + 1) if s not in self.bases]
        if len(free) < 2:
            return [self.label_logprobs(seg, labels) for seg, labels in items]
        groups = {}
        for i, item in enumerate(items):
            s1 = item[0][1]
            if s1 not in groups:
                groups[s1] = []
            groups[s1].append(i)
        shared_groups = {s1: idxs for s1, idxs in groups.items() if len(idxs) > 1}
        single_groups = {s1: idxs for s1, idxs in groups.items() if len(idxs) == 1}
        n_blocks = min(len(shared_groups), len(free) // 2)
        sorted_shared = sorted(shared_groups.items(), key=lambda x: len(x[1]), reverse=True)
        block_seq = {s1: free[j] for j, (s1, _) in enumerate(sorted_shared[:n_blocks])}
        tails = free[n_blocks:]
        out = [None] * len(items)
        paid = set()
        first = True
        for s1, seq in block_seq.items():
            p1 = self._tokens(s1)
            self.seq_rm(seq, -1, -1)
            self.seq_cp(seq_b, seq)
            self._decode(p1, n0, seq)
        jobs = []
        for i, item in enumerate(items):
            s1, s2 = item[0][1], item[0][2]
            if s1 in block_seq:
                src = block_seq[s1]
                pos = n0 + len(self._tokens(s1))
                toks = self._tokens(s2 + self._wrap[1])
                block_len = len(self._tokens(s1))
            else:
                src = seq_b
                pos = n0
                toks = self._tokens(s1) + self._tokens(s2 + self._wrap[1])
                block_len = 0
            jobs.append((i, src, pos, toks, block_len))
        k = 0
        while k < len(jobs):
            chunk = []
            used = 0
            while k < len(jobs) and len(chunk) < len(tails):
                i, src, pos, toks, block_len = jobs[k]
                if len(toks) > self.n_batch:
                    raise ValueError("question longer than n_batch tokens")
                if used + len(toks) > self.n_batch:
                    break
                chunk.append((k, i, src, pos, toks, block_len))
                used += len(toks)
                k += 1
            if not chunk:
                break
            b = self.batch
            b.n_tokens = 0
            for c, (k_idx, i, src, pos, toks, block_len) in enumerate(chunk):
                seq = tails[c]
                self.seq_rm(seq, -1, -1)
                self.seq_cp(src, seq)
                for t, tok in enumerate(toks):
                    b.token[b.n_tokens] = tok
                    b.pos[b.n_tokens] = pos + t
                    b.n_seq_id[b.n_tokens] = 1
                    b.seq_id[b.n_tokens][0] = seq
                    b.logits[b.n_tokens] = int(t == len(toks) - 1)
                    b.n_tokens += 1
            rc = self.L.llama_decode(self.ctx, b)
            if rc != 0:
                raise RuntimeError(f"llama_decode failed with code {rc}")
            logits = np.ctypeslib.as_array(self.L.llama_get_logits(self.ctx), shape=(len(chunk), self.n_vocab))
            for c, (k_idx, i, src, pos, toks, block_len) in enumerate(chunk):
                row = logits[c].astype(np.float64)
                m = row.max()
                lp = row - (m + np.log(np.sum(np.exp(row - m))))
                labels_dict = {}
                labels = items[i][1]
                for lab in labels:
                    ids = {self.tokenize(v)[0] for v in (lab, " " + lab)}
                    labels_dict[lab] = float(math.log(sum(math.exp(lp[t_id]) for t_id in ids)))
                computed = len(toks)
                s1 = items[i][0][1]
                if s1 in block_seq and s1 not in paid:
                    computed += block_len
                    paid.add(s1)
                if first:
                    computed += made
                    first = False
                p1len = len(self._tokens(s1))
                p2len = len(self._tokens(items[i][0][2] + self._wrap[1]))
                st = {"input_tokens": computed, "cached_tokens": n0 + p1len + p2len - computed,
                      "output_tokens": 1}
                out[i] = (labels_dict, st)
            for c in range(len(chunk)):
                self.seq_rm(tails[c], -1, -1)
        for s1, seq in block_seq.items():
            self.seq_rm(seq, -1, -1)
        return out


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
