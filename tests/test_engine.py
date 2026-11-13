"""Integration test with a real GGUF model through the llama.cpp engine.

Skipped unless the environment variable KVINDEX_TEST_MODEL names a model that Ollama has downloaded (for example
`KVINDEX_TEST_MODEL=qwen3:0.6b pytest tests/test_engine.py`). It repeats the correctness checks of
benchmarks/checks.py: the logits computed through the cache equal the logits computed without it, on every
architecture, and resident prefixes do not leak into each other.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("KVINDEX_TEST_MODEL")


@pytest.mark.skipif(not MODEL, reason="set KVINDEX_TEST_MODEL to a GGUF model name to run the engine test")
def test_cache_matches_recomputation():
    result = subprocess.run([
        sys.executable,
        os.path.join(HERE, "..", "benchmarks", "checks.py"),
        "--llm", MODEL,
        "--kv-type", "q8_0"
    ], capture_output=True, text=True, timeout=1800, env=dict(os.environ, PYTHONIOENCODING="utf8"))
    lines = [l for l in result.stdout.splitlines() if l.startswith("[")]
    assert result.returncode == 0, "\n".join(lines) + result.stderr[-2000:]
    pass_lines = [l for l in lines if l.startswith("[PASS]")]
    fail_lines = [l for l in lines if l.startswith("[FAIL]")]
    assert len(pass_lines) >= 4, f"Expected at least 4 PASS lines, got {len(pass_lines)}"
    assert len(fail_lines) == 0, "\n".join(fail_lines)
