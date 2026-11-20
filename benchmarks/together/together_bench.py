"""Prompt orders compared on the same engine: options (1.2, parallel), state (prefix tree), together, together-lists,
together-both.

    python together_bench.py <model> <out.json> <orders comma separated> [typed: yes/no] [n_ctx] [resident]

The results in this folder were measured on branch `together` (commit a391826), where order="lists" of DecideKV 1.3
is called "together-lists" and the two variants left out of the library ("together": every question and its options
first, then only "answer to question k"; "together-both": the same with the question repeated after the state) also
exist. On main, run it with the orders options, state, lists.

A. One turn of 15 questions (4 Score, 11 Noul) about a short and a long state: median ms of 3 turns after a first
   one, tokens computed and reused in the last turn.
B. Typed Decisions (test, 2,000 decisions), all the questions of a case in one decide: accuracy overall and by type,
   paired difference against the first order of the list (bootstrap over decisions, U = 2 u), decisions right only
   in one of the two, median ms per case.
"""
import json
import os
import statistics
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
import decidekv  # noqa: E402
from data import gold_key, to_text, typed_decisions  # noqa: E402
from decidekv import KV, Noul, Score  # noqa: E402
from engine_bench import STATE  # noqa: E402

QS = {f"s{i}": Score(instructions=f"How strong is quality {i} of the scene?",
                     criteria=["none", "weak", "medium", "strong"]) for i in range(4)}
QS.update({f"n{i}": Noul(instructions=f"Is condition number {i} true in this scene?") for i in range(11)})


def key_of(ans):
    if ans.type == "choice":
        return ans.choice
    if ans.type == "score":
        return max(ans.probabilities, key=ans.probabilities.get)
    return "true" if ans.noul >= 0.5 else "false"


def turn(kv, text, order, repeats=4):
    ms, r = [], None
    for i in range(repeats):
        t0 = time.perf_counter()
        r = kv.decide(f"Turn {i + 1}.\n{text}", QS, order=order)
        ms.append((time.perf_counter() - t0) * 1000)
    return {"ms": round(statistics.median(ms[1:]), 1), "all_ms": [round(x, 1) for x in ms],
            "computed_tokens": r.usage.input_tokens, "reused_tokens": r.usage.cached_tokens,
            "answers": {q: key_of(a) for q, a in r.answers.items()}}


def typed(kv, test, order):
    hits, types, ms = [], [], []
    for c in test:
        t0 = time.perf_counter()
        r = kv.decide(to_text(c["state"]), c["questions"], order=order)
        ms.append((time.perf_counter() - t0) * 1000)
        for qid, g in c["gold"].items():
            hits.append(int(key_of(r.answers[qid]) == gold_key(g)))
            types.append(g["type"])
    return np.array(hits), types, ms


def boot_u(x, rng):
    return 2 * float(np.std([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)], ddof=1))


def main():
    warnings.simplefilter("ignore")
    model, path, orders = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    do_typed = len(sys.argv) <= 4 or sys.argv[4] != "no"
    n_ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 16384
    kw = {"resident": int(sys.argv[6])} if len(sys.argv) > 6 else {}
    print("decidekv from", decidekv.__file__, flush=True)
    long_state = " ".join(to_text(c["state"]) for c in typed_decisions("test")[:9])
    res = {"model": model, "orders": orders, "n_ctx": n_ctx, "kw": kw}
    kv = KV(model, n_ctx=n_ctx, kv_type="q8_0", **kw)
    try:
        for sname, state in (("short", STATE), ("long", long_state)):
            text = state if isinstance(state, str) else to_text(state)
            for order in orders:
                res[f"turn/{sname}/{order}"] = turn(kv, text, order)
                print(f"turn {sname} {order}: {res[f'turn/{sname}/{order}']['ms']} ms, computed "
                      f"{res[f'turn/{sname}/{order}']['computed_tokens']} tokens", flush=True)
        if do_typed:
            test = typed_decisions("test")
            rng = np.random.default_rng(0)
            base = None
            for order in orders:
                typed(kv, test[:3], order)
                h, types, ms = typed(kv, test, order)
                row = {"acc": float(h.mean()), "U": boot_u(h, rng), "ms_case": round(statistics.median(ms), 1),
                       "ms_total_s": round(sum(ms) / 1000, 1),
                       "by_type": {t: float(np.mean([x for x, tt in zip(h, types) if tt == t])) for t in set(types)}}
                if base is None:
                    base = (order, h)
                else:
                    d = h - base[1]
                    row.update({"vs": base[0], "diff": float(d.mean()), "U_diff": boot_u(d, rng),
                                "only_this_right": int(((h == 1) & (base[1] == 0)).sum()),
                                "only_base_right": int(((h == 0) & (base[1] == 1)).sum())})
                print(f"typed {order}: {json.dumps(row)}", flush=True)
                row["hits"] = h.tolist()
                res[f"typed/{order}"] = row
                json.dump(res, open(path, "w"), indent=1)
    finally:
        kv.close()
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
