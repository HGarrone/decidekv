"""
The resident prefix of the llama.cpp engine, measured on any model (also models with recurrent layers).

For each model: Banking77 decisions with the option list resident in the cache against the same decisions with the
whole prompt recomputed every time, and one turn of 15 typed questions (4 Score, 11 Noul) about one state.

    python benchmarks/engine_bench.py --models qwen3:8b,hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ2_S
"""
import argparse
import os
import statistics
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import banking77, correct, load_json, save_json, to_text, typed_decisions  # noqa: E402
from kvindex import KV, Noul, Score  # noqa: E402

PROMPT = dict(noun="intent", state_label="Customer message",
              question="Which intent does this customer message express?")
STATE = {"scene": "The actor stops mid-line, looks at the audience and laughs nervously. The other actors wait."}


def label(model):
    return model.replace("hf.co/", "").replace(":", "_").replace("/", "_")


def banking(kv, options, queries, recompute):
    """Accuracy and median ms per decision; with `recompute`, the cache is emptied before every decision."""
    idx = kv.index(options, **PROMPT)
    be = kv.client
    ok, ms = 0, []
    for message, gold in queries:
        if recompute:
            be.clear()
            be._registered.clear()
        t0 = time.perf_counter()
        a = idx.choose(message)
        t1 = time.perf_counter()
        ms.append((t1 - t0) * 1000)
        ok += correct(a.choice, gold)
        if recompute:
            be._registered[idx.prefix] = None
    return {"acc": ok / len(queries), "ms": statistics.median(ms), "queries": len(queries)}


def turn(kv, state, order, repeats=4):
    """Median ms of a turn of 15 typed questions (4 Score, 11 Noul) about one state,
    in one prompt order, and its token counts. The first turn (which fills the cache
    with the option lists) is not counted."""
    qs = {f"s{i}": Score(instructions=f"How strong is quality {i} of the scene?",
                         criteria=["none", "weak", "medium", "strong"]) for i in range(4)}
    qs.update({f"n{i}": Noul(instructions=f"Is condition number {i} true in this scene?") for i in range(11)})
    ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = kv.decide(state, qs, order=order)
        t1 = time.perf_counter()
        ms.append((t1 - t0) * 1000)
    return {"ms": statistics.median(ms[1:]), "questions": 15, "computed_tokens": r.usage.input_tokens,
            "reused_tokens": r.usage.cached_tokens, "state_chars": len(str(state)), "order": order}


def main():
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen3:8b", help="Comma separated list of models")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries for banking77")
    parser.add_argument("--n-ctx", type=int, default=8192, help="Context size")
    parser.add_argument("--kv-type", default="q8_0", help="KV cache type")
    parser.add_argument("--out", default="engine_qwen3_8b.json", help="Output JSON file")
    a = parser.parse_args()

    res = load_json(a.out)
    options, queries = banking77(a.queries)
    long_state = " ".join(to_text(c["state"]) for c in typed_decisions("test")[:9])

    for model in a.models.split(","):
        kv = KV(model, n_ctx=a.n_ctx, kv_type=a.kv_type)
        try:
            key = label(model)
            res[f"{key}/info"] = {"architecture": kv.client.architecture(), "tail_removal": kv.client.can_cut}
            res[f"{key}/banking77/resident"] = banking(kv, options, queries, False)
            res[f"{key}/banking77/recompute"] = banking(kv, options, queries, True)
            for sname, state in (("short", STATE), ("long", long_state)):
                for order in ("options", "auto"):
                    res[f"{key}/turn15/{sname}/{order}"] = turn(kv, state, order)
            res[f"{key}/turn15"] = res[f"{key}/turn15/short/auto"]
            save_json(a.out, res)
            for k in [k for k in res if k.startswith(key + "/")]:
                print(f"{k}: {res[k]}", flush=True)
        finally:
            kv.close()


if __name__ == "__main__":
    main()
