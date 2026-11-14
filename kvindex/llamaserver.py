"""llama-server engine: the same backend contract as the in-process engine, over HTTP.

Each resident prefix (an option list, or a state shared by several questions) gets a server slot of its own, so the
server keeps its KV cache between calls (`cache_prompt`); one slot is kept for the other prompts. Start the server
with several slots (`-np`), e.g.

    llama-server -m model.gguf -c 32768 -np 8 -fa on -ctk q8_0 -ctv q8_0

or let `LlamaServer.start` launch the llama.cpp build found by kvindex. `score_options` (`full=True`) needs the logits
of arbitrary tokens and is only available with the in-process engine.
"""
from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections import OrderedDict


class LlamaServer:
    """Backend that talks to a llama-server (see the module docstring)."""

    def __init__(self, url: str = "http://127.0.0.1:8080", timeout: float = 600, process=None):
        self.host = url.rstrip("/")
        self.timeout = timeout
        self.process = process
        props = self._get("/props")
        self.slots = int(props.get("total_slots", 1))
        self.model_path = props.get("model_path", "")
        self.n_ctx = int(self._get("/slots")[0].get("n_ctx", 0))
        self._wrap = self._chat_wrap(props.get("chat_template") or "", props.get("bos_token") or "")
        self._slot_of = OrderedDict()
        self.scratch = self.slots - 1

    @classmethod
    def start(cls, model: str, port: int = 8080, n_ctx: int = 32768, slots: int = 8, kv_type: str = "q8_0",
              n_gpu_layers: int = 99, exe: str | None = None, wait: float = 300) -> "LlamaServer":
        """Launch llama-server on a GGUF model (an Ollama model name or a .gguf path) and connect to it."""
        from .llamacpp import ollama_gguf
        path = model if model.endswith(".gguf") else ollama_gguf(model)
        exe = exe or _find_server()
        if not exe:
            raise FileNotFoundError("llama-server not found: install a llama.cpp build in ~/.kvindex or ./vendor, "
                                   "or pass exe=")
        cmd = [exe, "-m", path, "--port", str(port), "-c", str(n_ctx), "-np", str(slots), "-ngl", str(n_gpu_layers),
               "-fa", "on", "-ctk", kv_type, "-ctv", kv_type]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{port}"
        deadline = time.time() + wait
        while time.time() < deadline:
            code = proc.poll()
            if code is not None:
                raise RuntimeError(f"llama-server exited with code {code}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=5) as resp:
                    ready = json.load(resp).get("status") == "ok"
            except Exception:
                ready = False
            if ready:
                return cls(url, process=proc)
            time.sleep(0.5)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise TimeoutError("llama-server did not become ready")

    def close(self) -> None:
        """Stop the server if this object started it."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def _get(self, path: str):
        req = urllib.request.Request(self.host + path)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(body).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"llama-server error {e.code}: {e.read().decode(errors='replace')}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"llama-server not reachable at {self.host} ({e.reason})") from None

    @staticmethod
    def _chat_wrap(template: str, bos: str) -> tuple[str, str]:
        """Text before and after the user message in the model's chat template (thinking disabled)."""
        if not template:
            return "", ""
        import jinja2
        marker = "\x00KVINDEX\x00"
        out = jinja2.Environment().from_string(template).render(
            messages=[{"role": "user", "content": marker}],
            add_generation_prompt=True,
            enable_thinking=False,
            bos_token=bos,
            eos_token=""
        )
        before, after = out.split(marker)
        if bos and before.startswith(bos):
            before = before[len(bos):]
        return before, after

    def architecture(self) -> str:
        return ""

    def _prompt(self, segments) -> str:
        return self._wrap[0] + "".join(segments) + self._wrap[1]

    def _register(self, prefix: str) -> None:
        """Give `prefix` a slot of its own (the least recently used one when all are taken)."""
        if prefix in self._slot_of:
            self._slot_of.move_to_end(prefix)
            return
        usable = max(1, self.slots - 1)
        taken = set(self._slot_of.values())
        if len(self._slot_of) >= usable:
            _, slot = self._slot_of.popitem(last=False)
        else:
            slot = next(s for s in range(usable) if s not in taken)
        self._slot_of[prefix] = slot

    def _slot(self, segments) -> int:
        slot = self._slot_of.get(segments[0])
        if slot is None:
            return self.scratch
        self._slot_of.move_to_end(segments[0])
        return slot

    def _complete(self, segments, **body) -> dict:
        req = {"prompt": self._prompt(segments), "cache_prompt": True, "id_slot": self._slot(segments),
               "temperature": 0}
        req.update(body)
        return self._post("/completion", req)

    @staticmethod
    def _stats(j: dict) -> dict:
        t = j.get("timings") or {}
        return {"input_tokens": int(t.get("prompt_n", 0)), "cached_tokens": int(t.get("cache_n", 0)),
                "output_tokens": int(j.get("tokens_predicted", 0))}

    def prefill(self, segments, num_ctx=None) -> dict:
        """Declare the first segment a resident prefix and evaluate the prompt once."""
        self._register(segments[0])
        return self._stats(self._complete(segments, n_predict=0))

    def hold(self, prefix: str) -> None:
        """Declare `prefix` a resident prefix (see `LlamaCpp.hold`)."""
        self._register(prefix)

    def release(self, prefix: str) -> None:
        """Forget a resident prefix declared with `hold`."""
        self._slot_of.pop(prefix, None)

    def label_logprobs(self, segments, labels, num_ctx=None):
        """log P of each one-token label (tokens equal to the label up to spaces are summed); -> (dict, stats)."""
        j = self._complete(segments, n_predict=1, n_probs=100)
        top = (j.get("completion_probabilities") or [{}])[0].get("top_logprobs", [])
        out = {}
        for lab in labels:
            ps = [math.exp(t["logprob"]) for t in top if t["token"].strip() == lab]
            out[lab] = math.log(sum(ps)) if ps else -30.0
        st = self._stats(j)
        st["output_tokens"] = 1
        return out, st

    def pick_option(self, segments, options, num_ctx=None, enum_max=None, allowed=None):
        """Greedy decoding constrained to the option names (a GBNF grammar); -> (option, log-prob, stats)."""
        pool = list(allowed) if allowed is not None else list(options)
        grammar = "root ::= " + " | ".join(_gbnf_literal(o) for o in pool)
        j = self._complete(segments, n_predict=max(len(o) for o in pool) + 8, n_probs=1, grammar=grammar)
        toks = [t for t in (j.get("completion_probabilities") or []) if t.get("token")]
        total = sum(float(t["logprob"]) for t in toks)
        return j.get("content", ""), total, self._stats(j)

    def score_options(self, segments, options, allowed=None, floor: float = 1e-6):
        raise NotImplementedError("full=True needs the in-process engine (backend='llamacpp')")


def _gbnf_literal(text: str) -> str:
    """`text` as a GBNF string literal."""
    return json.dumps(text)


def _find_server() -> str | None:
    """llama-server of a llama.cpp build in KVINDEX_LLAMA_LIB, ~/.kvindex or ./vendor (the newest), or None."""
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    lib = os.environ.get("KVINDEX_LLAMA_LIB")
    if lib and os.path.exists(os.path.join(lib, name)):
        return os.path.join(lib, name)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = sorted(glob.glob(os.path.join(os.path.expanduser("~"), ".kvindex", "llama.cpp-*", "bin", name)) +
                   glob.glob(os.path.join(here, "vendor", "llama.cpp-*", "bin", name)))
    return found[-1] if found else None
