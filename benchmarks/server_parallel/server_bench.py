"""Label questions on llama-server: one at a time (DecideKV 1.3) against several slots at the same time (1.4).

    python server_bench.py <model> <out.json> [slots] [typed: yes / no / order/mode,order/mode,...]

Starts llama-server on the model (the llama.cpp build found by decidekv) with `slots` slots (default 8) and an 8-bit
KV cache. A. One turn of 15 questions (4 Score, 11 Noul) about a short and a long state, orders options / state /
lists, sequential and parallel: median ms of 3 turns after a first one. B. Typed Decisions (test, 2,000 decisions):
options sequential, options parallel, state parallel, lists parallel; accuracy, paired difference against options
sequential (bootstrap over decisions, U = 2 u), decisions right only in one of the two, median ms per case.
"""
import json
import os
import statistics
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
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


def turn(kv, text, order, tag, repeats=4):
    """Median ms of the last repeats-1 turns; `tag` makes the texts new to the server (no cache from another run)."""
    ms = []
    for i in range(repeats):
        t0 = time.perf_counter()
        kv.decide(f"Turn {i + 1} of run {tag}.\n{text}", QS, order=order)
        ms.append((time.perf_counter() - t0) * 1000)
    return {"ms": round(statistics.median(ms[1:]), 1), "all_ms": [round(x, 1) for x in ms]}


def typed(kv, test, order):
    hits, ms = [], []
    for c in test:
        t0 = time.perf_counter()
        r = kv.decide(to_text(c["state"]), c["questions"], order=order)
        ms.append((time.perf_counter() - t0) * 1000)
        for qid, g in c["gold"].items():
            hits.append(int(key_of(r.answers[qid]) == gold_key(g)))
    return np.array(hits), ms


def boot_u(x, rng):
    return 2 * float(np.std([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)], ddof=1))


def main():
    warnings.simplefilter("ignore")
    model, path = sys.argv[1], sys.argv[2]
    slots = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    spec = sys.argv[4] if len(sys.argv) > 4 else "yes"
    do_typed = spec != "no"
    runs = [("options", False), ("options", True), ("state", True), ("lists", True)] if spec in ("yes", "no") else         [(x.split("/")[0], x.split("/")[1] == "parallel") for x in spec.split(",")]
    print("decidekv", decidekv.__version__, "from", decidekv.__file__, flush=True)
    long_state = " ".join(to_text(c["state"]) for c in typed_decisions("test")[:9])
    res = {"model": model, "slots": slots, "engine": "llama-server"}
    kv = KV(model, backend="llamaserver", n_ctx=4096 * slots, slots=slots, port=8091)
    try:
        for sname, state in (("short", STATE), ("long", long_state)):
            text = state if isinstance(state, str) else to_text(state)
            for order in ("options", "state", "lists"):
                for par in (False, True):
                    kv.client.parallel = par
                    key = f"turn/{sname}/{order}/{'parallel' if par else 'sequential'}"
                    res[key] = turn(kv, text, order, key)
                    print(key, res[key]["ms"], "ms", flush=True)
        if do_typed:
            test = typed_decisions("test")
            rng = np.random.default_rng(0)
            base = None
            for order, par in runs:
                kv.client.parallel = par
                typed(kv, test[:3], order)
                h, ms = typed(kv, test, order)
                name = f"{order}/{'parallel' if par else 'sequential'}"
                row = {"acc": float(h.mean()), "U": boot_u(h, rng), "ms_case": round(statistics.median(ms), 1)}
                if base is None:
                    base = h
                else:
                    d = h - base
                    row.update({"diff": float(d.mean()), "U_diff": boot_u(d, rng),
                                "only_this_right": int(((h == 1) & (base == 0)).sum()),
                                "only_base_right": int(((h == 0) & (base == 1)).sum())})
                print(f"typed {name}: {json.dumps(row)}", flush=True)
                row["hits"] = h.tolist()
                res[f"typed/{name}"] = row
                json.dump(res, open(path, "w"), indent=1)
    finally:
        kv.close()
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
