"""Typed Decisions (test), all the questions of a case in one decide, options first: one by one vs parallel.

    python typed_ab.py <model> <out.json> [resident]
"""
import json
import statistics
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, r"E:\AI\PAI_1shot_opt")
sys.path.insert(0, r"E:\AI\PAI_1shot_opt\benchmarks")
from data import gold_key, to_text, typed_decisions  # noqa: E402
from decidekv import KV  # noqa: E402


def key_of(ans):
    if ans.type == "choice":
        return ans.choice
    if ans.type == "score":
        return max(ans.probabilities, key=ans.probabilities.get)
    return "true" if ans.noul >= 0.5 else "false"


def run(kv, test, batch):
    kv.batch_options = batch
    hits, ms = [], []
    for c in test:
        t0 = time.perf_counter()
        r = kv.decide(to_text(c["state"]), c["questions"], order="options")
        ms.append((time.perf_counter() - t0) * 1000)
        for qid, g in c["gold"].items():
            hits.append(int(key_of(r.answers[qid]) == gold_key(g)))
    return np.array(hits), ms


def main():
    warnings.simplefilter("ignore")
    model, path = sys.argv[1], sys.argv[2]
    kw = {"resident": int(sys.argv[3])} if len(sys.argv) > 3 else {}
    test = typed_decisions("test")
    kv = KV(model, n_ctx=16384, kv_type="q8_0", **kw)
    try:
        run(kv, test[:3], False)
        a, ms_a = run(kv, test, False)
        b, ms_b = run(kv, test, True)
    finally:
        kv.close()
    rng = np.random.default_rng(0)
    d = b - a
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    boot_a = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(2000)]
    res = {"model": model, "kw": kw, "decisions": len(a), "cases": len(test),
           "acc_one_by_one": a.mean(), "acc_parallel": b.mean(), "U_acc": 2 * float(np.std(boot_a, ddof=1)),
           "diff": d.mean(), "U_diff": 2 * float(np.std(boot, ddof=1)),
           "only_one_by_one_right": int(((a == 1) & (b == 0)).sum()), "only_parallel_right": int(((b == 1) & (a == 0)).sum()),
           "ms_case_one_by_one": statistics.median(ms_a), "ms_case_parallel": statistics.median(ms_b),
           "ms_total_one_by_one": sum(ms_a), "ms_total_parallel": sum(ms_b)}
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
