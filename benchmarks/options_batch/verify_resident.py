"""Options first, one question at a time vs in parallel sequences (label_logprobs_resident): same answers, times.

    python verify_resident.py <model> <out.json> [resident]
"""
import json
import statistics
import os
import sys
import time
import warnings

sys.path.insert(0, r"E:\AI\PAI_1shot_opt")
sys.path.insert(0, r"E:\AI\PAI_1shot_opt\benchmarks")
import decidekv  # noqa: E402
from decidekv import KV, Noul, Score  # noqa: E402
from data import to_text, typed_decisions  # noqa: E402
from engine_bench import STATE  # noqa: E402

QS = {f"s{i}": Score(instructions=f"How strong is quality {i} of the scene?",
                     criteria=["none", "weak", "medium", "strong"]) for i in range(4)}
QS.update({f"n{i}": Noul(instructions=f"Is condition number {i} true in this scene?") for i in range(11)})


def probs(answer):
    if answer.type == "noul":
        return {"Yes": answer.noul}
    return dict(answer.probabilities)


def run(kv, text, order, batch, repeats=4):
    kv.batch_options = batch
    ms, out = [], None
    for i in range(repeats):
        t0 = time.perf_counter()
        r = kv.decide(f"Turn {i + 1}.\n{text}", QS, order=order)
        ms.append((time.perf_counter() - t0) * 1000)
        if i == repeats - 1:
            out = r
    return statistics.median(ms[1:]), ms, out


def main():
    warnings.simplefilter("ignore")
    model, path = sys.argv[1], sys.argv[2]
    kw = {"resident": int(sys.argv[3])} if len(sys.argv) > 3 else {}
    print("decidekv from", decidekv.__file__, flush=True)
    long_state = " ".join(to_text(c["state"]) for c in typed_decisions("test")[:9])
    kv = KV(model, n_ctx=8192, kv_type="q8_0", **kw)
    res = {"model": model, "kw": kw}
    try:
        for sname, state in (("short", STATE), ("long", long_state)):
            text = state if isinstance(state, str) else to_text(state)
            ref_ms, ref_all, ref = run(kv, text, "options", False)
            new_ms, new_all, new = run(kv, text, "options", True)
            auto_ms, auto_all, auto = run(kv, text, "auto", True)
            diff = max(abs(probs(ref.answers[q]).get(k, 0.0) - v) for q in QS for k, v in probs(new.answers[q]).items())
            same_top = sum(max(probs(ref.answers[q]), key=probs(ref.answers[q]).get)
                           == max(probs(new.answers[q]), key=probs(new.answers[q]).get) for q in QS if q[0] == "s")
            nouls = sum((ref.answers[q].noul >= 0.5) == (new.answers[q].noul >= 0.5) for q in QS if q[0] == "n")
            res[sname] = {
                "state_chars": len(text),
                "options_one_by_one_ms": round(ref_ms, 1), "options_parallel_ms": round(new_ms, 1),
                "auto_tree_ms": round(auto_ms, 1), "all_ms": [ref_all, new_all, auto_all],
                "max_prob_diff": diff, "score_top_agree": f"{same_top}/4", "noul_side_agree": f"{nouls}/11",
                "tokens_one_by_one": [ref.usage.input_tokens, ref.usage.cached_tokens],
                "tokens_parallel": [new.usage.input_tokens, new.usage.cached_tokens],
                "question_ms_parallel": new.question_ms}
            print(sname, {k: v for k, v in res[sname].items() if k not in ("all_ms", "question_ms_parallel")},
                  flush=True)
    finally:
        kv.close()
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
