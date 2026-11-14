"""
Optional catalogue training, measured on Banking77 (77 intents, public train/test split).

The embedding catalogue is trained with `KV.train` on the training messages and evaluated on test messages, with and
without the trained head: the catalogue alone (k=1, no LLM call) and the catalogue followed by the LLM (the k closest
intents in the prompt). A second training uses the examples of half of the intents only, to measure what happens to
intents that had no training example.

    python benchmarks/train_bench.py
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import banking77, banking77_train, correct, load_json, save_json  # noqa: E402
from kvindex import KV  # noqa: E402
from kvindex.catalog import TopKIndex  # noqa: E402

PROMPT = dict(noun="intent", state_label="Customer message",
              question="Which intent does this customer message express?")


def measure(idx, queries):
    """Accuracy and median time per decision (ms) of an index on (message, gold) queries."""
    times = []
    hits = 0
    for message, gold in queries:
        t0 = time.perf_counter()
        ans = idx.choose(message)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        if correct(ans.choice, gold):
            hits += 1
    return {"acc": hits / len(queries), "ms": statistics.median(times), "queries": len(queries)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="qwen3:8b")
    parser.add_argument("--embed", default="qwen3-embedding:4b")
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--kv-type", default="q8_0")
    a = parser.parse_args()

    out_name = f"train_{a.llm.replace(':', '_')}.json"
    res = load_json(out_name)

    options, queries = banking77(a.queries)
    pairs = banking77_train()

    def log(name, row):
        res[f"banking77/{name}"] = row
        save_json(out_name, res)
        print(f"{name:<24} {row}", flush=True)

    kv = KV(a.llm, embed=a.embed, n_ctx=a.n_ctx, kv_type=a.kv_type)
    try:
        log("flat", measure(kv.index(options, **PROMPT), queries))

        t0 = time.perf_counter()
        head = kv.train(options, pairs, question=PROMPT["question"])
        log("training", {"examples": len(pairs), "total_s": time.perf_counter() - t0, **head.report})

        for k in (1, 5, 50):
            log(f"catalogue-k{k}", measure(TopKIndex(kv, options, kv.encoder, k=k, **PROMPT), queries))
            log(f"catalogue-k{k}-trained", measure(kv.index(options, k=k, head=head, **PROMPT), queries))

        seen = set(options[::2])
        half_pairs = [p for p in pairs if p[1] in seen]
        t0 = time.perf_counter()
        half = kv.train(options, half_pairs, question=PROMPT["question"])
        log("half-training", {"examples": len(half_pairs), "total_s": time.perf_counter() - t0, **half.report})

        groups = {
            "seen": [q for q in queries if next(iter(q[1])) in seen],
            "unseen": [q for q in queries if next(iter(q[1])) not in seen],
        }
        for name, subset in groups.items():
            log(f"half-{name}", measure(TopKIndex(kv, options, kv.encoder, k=1, **PROMPT), subset))
            idx = kv.index(options, k=1, head=half, **PROMPT)
            before = idx.fallbacks
            row = measure(idx, subset)
            row["fallback_rate"] = (idx.fallbacks - before) / len(subset)
            log(f"half-{name}-trained", row)
            off = kv.index(options, k=1, head=half, fallback=0, **PROMPT)
            log(f"half-{name}-trained-nofallback", measure(off, subset))
    finally:
        kv.close()


if __name__ == "__main__":
    main()
