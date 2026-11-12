"""Tiny Ollama client (standard library only)."""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request

# architectures with recurrent / state-space layers: Ollama cannot reuse a partial prefix for them
HYBRID_ARCHS = ("qwen35", "qwen3next", "mamba", "jamba", "rwkv", "falcon_h1", "granitehybrid", "nemotron_h", "lfm2")


class OllamaError(RuntimeError):
    pass


def client_host(host: str | None = None) -> str:
    """Normalise OLLAMA_HOST the way the ollama CLI does: '0.0.0.0' -> localhost, default port 11434."""
    h = (host or os.environ.get("OLLAMA_HOST") or "127.0.0.1").strip().rstrip("/")
    scheme = "http"
    if "://" in h:
        scheme, h = h.split("://", 1)
    hostname, _, port = h.partition(":")
    if hostname in ("", "0.0.0.0", "::", "[::]"):
        hostname = "127.0.0.1"
    return f"{scheme}://{hostname}:{port or 11434}"


class Ollama:
    def __init__(self, model: str, host: str | None = None, keep_alive: str = "30m", timeout: float = 600):
        self.model = model
        self.host = client_host(host)
        self.keep_alive, self.timeout = keep_alive, timeout
        self._think = True                   # send think=false unless the model rejects the flag

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.host + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            try:
                msg = json.loads(msg).get("error", msg)
            except ValueError:
                pass
            raise OllamaError(msg) from None
        except urllib.error.URLError as e:
            raise OllamaError(f"Ollama not reachable at {self.host} ({e.reason}). Is `ollama serve` running?") from None

    def architecture(self) -> str:
        try:
            info = self._post("/api/show", {"model": self.model}).get("model_info", {})
            return str(info.get("general.architecture", ""))
        except OllamaError:
            return ""

    def generate(self, prompt: str, num_ctx: int, num_predict: int = 1, top_logprobs: int = 0,
                 enum: list[str] | None = None, stop: list[str] | None = None) -> dict:
        body = {"model": self.model, "prompt": prompt, "stream": False, "keep_alive": self.keep_alive,
                "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict}}
        if self._think:
            body["think"] = False
        if top_logprobs:
            body["logprobs"], body["top_logprobs"] = True, min(20, top_logprobs)
        if enum is not None:
            body["format"] = {"type": "string", "enum": enum}
        if stop:
            body["options"]["stop"] = stop
        try:
            return self._post("/api/generate", body)
        except OllamaError as e:
            if self._think and "think" in str(e).lower():
                self._think = False
                return self.generate(prompt, num_ctx, num_predict, top_logprobs, enum, stop)
            raise

    # ------------------------------------------------------------ backend contract
    @staticmethod
    def _stats(j: dict) -> dict:
        total = j.get("prompt_eval_count", 0) or 0
        cached = j.get("prompt_eval_cached_count", 0) or 0
        return {"input_tokens": total - cached, "cached_tokens": cached, "output_tokens": j.get("eval_count", 0) or 0}

    def prefill(self, segments, num_ctx):
        """Evaluate the prompt once (fills the server's prefix cache); -> stats with the exact token count."""
        return self._stats(self.generate("".join(segments), num_ctx, num_predict=1))

    def label_logprobs(self, segments, labels, num_ctx):
        """-> ({label: logprob} for the labels visible in the top-20, stats). One forward pass."""
        prompt = "".join(segments)
        j = self.generate(prompt, num_ctx, num_predict=1, top_logprobs=20)
        st = self._stats(j)
        lp: dict[str, float] = {}
        for t in (j.get("logprobs") or [{}])[0].get("top_logprobs", []):
            tok = t["token"].strip()
            if tok in labels:
                lp[tok] = math.log(math.exp(lp.get(tok, -1e30)) + math.exp(t["logprob"]))
        if not lp:          # the model answered off-format: force a valid label with the grammar
            j = self.generate(prompt, num_ctx, num_predict=4, enum=list(labels))
            for k, v in self._stats(j).items():
                st[k] += v
            lp = {json.loads(j["response"]): 0.0}
        return lp, st

    def pick_option(self, segments, options, num_ctx, enum_max=0, allowed=None):
        """-> (decoded option text, its logprob or None, stats). Grammar-constrained up to `enum_max` options;
        `allowed` restricts the grammar to a subset (excluded options become impossible to emit)."""
        pool = list(allowed) if allowed is not None else list(options)
        enum = pool if len(pool) <= enum_max else None
        j = self.generate("".join(segments), num_ctx, num_predict=48, top_logprobs=1, enum=enum,
                          stop=None if enum else ["\n"])
        text = j.get("response", "").strip()
        if enum:
            text = json.loads(text)
        lps = [t.get("logprob", 0.0) for t in (j.get("logprobs") or [])]
        return text, (sum(lps) if lps else None), self._stats(j)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._post("/api/embed", {"model": self.model, "input": texts, "keep_alive": self.keep_alive,
                                         "options": {"num_ctx": 512}})["embeddings"]
