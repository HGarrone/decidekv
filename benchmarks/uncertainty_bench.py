"""
Type A measurement uncertainty of DecideKV, accuracy and time, on three worst cases.

Every case is run `--runs` times on the same items. Two components are estimated, as in the GUM:
- repeatability: the standard deviation of the per-run accuracy (and of the per-run median time) over the runs;
- finite test set: the standard deviation of the accuracy (and of the median time) over bootstrap resamples of the
  items of the first run.
They are combined in quadrature (u_c) and expanded with k = 2 (U). For Typed Decisions the OpenJev architecture is run
on the same items, and the difference of accuracy gets a paired bootstrap uncertainty and a McNemar test.

    python benchmarks/uncertainty_bench.py --runs 5
"""
import argparse
import math
import os
import statistics
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines import LIST_PROMPTS, openjev_pick  # noqa: E402
from data import banking77, correct, gold_key, load_json, save_json, to_text, typed_decisions, wordnet  # noqa: E402
from decidekv import KV  # noqa: E402

M27 = "hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ2_S"
B = 2000


def answer_key(a):
    """The answer object as a key comparable with `gold_key`."""
    if a.type == "choice":
        return a.choice
    if a.type == "score":
        return max(a.probabilities, key=a.probabilities.get)
    if a.type == "noul":
        return "true" if a.noul >= 0.5 else "false"


def run_typed_decidekv(kv, test):
    """One run of DecideKV on Typed Decisions (the questions of a case in one decide call): per-item hits (gold
    decisions, in a fixed order) and per-decision ms (the case time divided by its questions, once per gold item)."""
    hits, ms = [], []
    for c in test:
        state = to_text(c["state"])
        start = time.perf_counter()
        r = kv.decide(state, c["questions"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        share = elapsed_ms / len(c["questions"])
        for qid, g in c["gold"].items():
            hits.append(answer_key(r.answers[qid]) == gold_key(g))
            ms.append(share)
    return hits, ms


def run_typed_openjev(kv, test):
    """One run of the OpenJev architecture on the same items: per-item hits and per-decision ms."""
    hits, ms = [], []
    for c in test:
        state = to_text(c["state"])
        for qid, g in c["gold"].items():
            qd = c["questions"][qid]
            start = time.perf_counter()
            q, ix, ins = kv.question_index(qd, warm=False)
            labels = ix.labels or ix.keys
            lab = openjev_pick(kv.client, ix, state, ins, labels)
            key = ix.keys[labels.index(lab)]
            if q.type == "noul":
                key = "true" if key == "Yes" else "false"
            elapsed = (time.perf_counter() - start) * 1000
            hits.append(key == gold_key(g))
            ms.append(elapsed)
    return hits, ms


def run_list(idx, queries):
    """One run of an index on (state, gold) queries: per-item hits and per-decision ms."""
    hits, ms = [], []
    for state, gold in queries:
        start = time.perf_counter()
        a = idx.choose(state)
        elapsed_ms = (time.perf_counter() - start) * 1000
        hits.append(bool(correct(a.choice, gold)))
        ms.append(elapsed_ms)
    return hits, ms


def components(runs):
    """Uncertainty components from a list of runs, each (hits, ms)."""
    rng = np.random.default_rng(0)
    accs = [float(np.mean(h)) for h, _ in runs]
    meds = [float(statistics.median(m)) for _, m in runs]
    u_rep_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
    u_rep_ms = statistics.stdev(meds) if len(meds) > 1 else 0.0
    h0 = np.asarray(runs[0][0], dtype=float)
    m0 = np.asarray(runs[0][1], dtype=float)
    n = len(h0)
    idx = rng.integers(0, n, size=(B, n))
    u_test_acc = float(np.std(h0[idx].mean(axis=1), ddof=1))
    u_test_ms = float(np.std(np.median(m0[idx], axis=1), ddof=1))
    u_c_acc = math.sqrt(u_rep_acc ** 2 + u_test_acc ** 2)
    u_c_ms = math.sqrt(u_rep_ms ** 2 + u_test_ms ** 2)
    identical = sum(1 for h, _ in runs[1:] if list(h) == list(runs[0][0]))
    return {
        "runs": len(runs),
        "items": n,
        "acc_runs": accs,
        "ms_runs": meds,
        "acc": statistics.mean(accs),
        "ms": statistics.mean(meds),
        "u_rep_acc": u_rep_acc,
        "u_test_acc": u_test_acc,
        "u_c_acc": u_c_acc,
        "U_acc": 2 * u_c_acc,
        "u_rep_ms": u_rep_ms,
        "u_test_ms": u_test_ms,
        "u_c_ms": u_c_ms,
        "U_ms": 2 * u_c_ms,
        "runs_identical_to_first": identical
    }


def paired(runs_a, runs_b):
    """Difference of accuracy a - b on the same items: paired bootstrap (the items are the same for both methods, so
    their test-set errors are correlated and largely cancel), repeatability of both, McNemar test."""
    rng = np.random.default_rng(1)
    a = np.asarray(runs_a[0][0], dtype=float)
    b = np.asarray(runs_b[0][0], dtype=float)
    n = len(a)
    idx = rng.integers(0, n, size=(B, n))
    # difference of the mean accuracies over the runs; the test-set component is estimated on the first run
    diff = float(np.mean([np.mean(h) for h, _ in runs_a]) - np.mean([np.mean(h) for h, _ in runs_b]))
    u_test = float(np.std(a[idx].mean(axis=1) - b[idx].mean(axis=1), ddof=1))
    rep_a = statistics.stdev([float(np.mean(h)) for h, _ in runs_a]) if len(runs_a) > 1 else 0.0
    rep_b = statistics.stdev([float(np.mean(h)) for h, _ in runs_b]) if len(runs_b) > 1 else 0.0
    u_c = math.sqrt(u_test ** 2 + rep_a ** 2 + rep_b ** 2)
    only_a = int(np.sum((a == 1) & (b == 0)))
    only_b = int(np.sum((a == 0) & (b == 1)))
    m = only_a + only_b
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0
    return {
        "diff": diff,
        "u_c": u_c,
        "U": 2 * u_c,
        "only_a": only_a,
        "only_b": only_b,
        "p_mcnemar": p
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cases", default="typed8b,banking27b,wordnet30k")
    parser.add_argument("--embed", default="qwen3-embedding:4b")
    a = parser.parse_args()
    warnings.simplefilter("ignore")
    out = "uncertainty.json"
    res = load_json(out)
    cases = a.cases.split(",")
    if "typed8b" in cases:
        test = typed_decisions("test")
        kv = KV("qwen3:8b", n_ctx=16384, kv_type="q8_0")
        try:
            dk = [run_typed_decidekv(kv, test) for _ in range(a.runs)]
            oj = [run_typed_openjev(kv, test) for _ in range(a.runs)]
            res["typed8b/decidekv"] = components(dk)
            res["typed8b/openjev"] = components(oj)
            res["typed8b/decidekv-openjev"] = paired(dk, oj)
            save_json(out, res)
            print(
                f"typed8b/decidekv: acc={res['typed8b/decidekv']['acc']:.3f}, "
                f"U_acc={res['typed8b/decidekv']['U_acc']:.3f}, "
                f"ms={res['typed8b/decidekv']['ms']:.1f}, "
                f"U_ms={res['typed8b/decidekv']['U_ms']:.1f}",
                flush=True
            )
            print(
                f"typed8b/openjev: acc={res['typed8b/openjev']['acc']:.3f}, "
                f"U_acc={res['typed8b/openjev']['U_acc']:.3f}, "
                f"ms={res['typed8b/openjev']['ms']:.1f}, "
                f"U_ms={res['typed8b/openjev']['U_ms']:.1f}",
                flush=True
            )
            print(
                f"typed8b/decidekv-openjev: diff={res['typed8b/decidekv-openjev']['diff']:.3f}, "
                f"U={res['typed8b/decidekv-openjev']['U']:.3f}, "
                f"p={res['typed8b/decidekv-openjev']['p_mcnemar']:.3f}",
                flush=True
            )
        finally:
            kv.close()
    if "banking27b" in cases:
        options, queries = banking77(300)
        kv = KV(M27, n_ctx=8192, kv_type="q8_0")
        try:
            idx = kv.index(options, **LIST_PROMPTS["banking77"])
            idx.choose(queries[0][0])
            runs = [run_list(idx, queries) for _ in range(a.runs)]
            res["banking27b/decidekv"] = components(runs)
            save_json(out, res)
            print(
                f"banking27b/decidekv: acc={res['banking27b/decidekv']['acc']:.3f}, "
                f"U_acc={res['banking27b/decidekv']['U_acc']:.3f}, "
                f"ms={res['banking27b/decidekv']['ms']:.1f}, "
                f"U_ms={res['banking27b/decidekv']['U_ms']:.1f}",
                flush=True
            )
        finally:
            kv.close()
    if "wordnet30k" in cases:
        options, queries = wordnet(30000, 600, core=1000)
        kv = KV("qwen3:8b", embed=a.embed, n_ctx=16384, kv_type="q8_0")
        try:
            idx = kv.index(options, k=50, **LIST_PROMPTS["wordnet"])
            idx.choose(queries[0][0])
            runs = [run_list(idx, queries) for _ in range(a.runs)]
            res["wordnet30k/decidekv"] = components(runs)
            save_json(out, res)
            print(
                f"wordnet30k/decidekv: acc={res['wordnet30k/decidekv']['acc']:.3f}, "
                f"U_acc={res['wordnet30k/decidekv']['U_acc']:.3f}, "
                f"ms={res['wordnet30k/decidekv']['ms']:.1f}, "
                f"U_ms={res['wordnet30k/decidekv']['U_ms']:.1f}",
                flush=True
            )
        finally:
            kv.close()


if __name__ == "__main__":
    main()
