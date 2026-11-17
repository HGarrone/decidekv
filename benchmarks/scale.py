"""
decidekv on growing option sets: WordNet reverse dictionary (definition -> word).

`KV.index` keeps the whole list in the KV cache up to `flat_max` options and switches to the embedding catalogue
(top-k options in the prompt) above it. For every size: accuracy, median time per decision, time to build the index.

    python benchmarks/scale.py --sizes 200,1000,3000,10000,30000
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import correct, load_json, save_json, wordnet  # noqa: E402
from decidekv import KV  # noqa: E402

PROMPT = dict(noun="word", state_label="Definition", question="Which word does this definition define?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="qwen3:8b")
    parser.add_argument("--embed", default="qwen3-embedding:4b")
    parser.add_argument("--sizes", default="200,1000,3000,10000,30000")
    parser.add_argument("--queries", type=int, default=600)
    parser.add_argument("--core", type=int, default=1000, help="queries are definitions of the first CORE entries")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--flat-max", type=int, default=3000)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--kv-type", default="q8_0")
    parser.add_argument("--redo", action="store_true")

    a = parser.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]

    out_name = f"scale_{a.llm.replace(':', '_')}.json"
    res = load_json(out_name)

    for n in sizes:
        key = f"{n}/q{a.queries}"
        if key in res and not a.redo:
            continue

        options, queries = wordnet(n, a.queries, core=a.core)
        kv = None
        try:
            kv = KV(a.llm, embed=a.embed, flat_max=a.flat_max, n_ctx=a.n_ctx, kv_type=a.kv_type)

            start = time.perf_counter()
            idx = kv.index(options, k=a.k, **PROMPT)
            build_s = time.perf_counter() - start

            times = []
            correct_count = 0
            recall = []

            for definition, gold in queries:
                start = time.perf_counter()
                ans = idx.choose(definition)
                end = time.perf_counter()
                times.append((end - start) * 1000)
                if correct(ans.choice, gold):
                    correct_count += 1
                candidates = getattr(idx, "last_candidates", None)
                if isinstance(candidates, list) and len(candidates) > 0:
                    recall.append(any(correct(c, gold) for c in candidates))

            acc = correct_count / len(queries)
            ms = statistics.median(times) if times else 0
            row = {"acc": acc, "ms": ms, "build_s": build_s,
                   "queries": len(queries), "index": type(idx).__name__,
                   "recall": sum(recall) / len(recall) if recall else None}
        except Exception as e:
            row = {"error": f"{type(e).__name__}: {e}"}
        finally:
            if kv is not None:
                kv.close()

        res[key] = row
        save_json(out_name, res)
        print(f"{n:>6} {row}", flush=True)


if __name__ == "__main__":
    main()
