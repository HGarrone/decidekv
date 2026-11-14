"""
Training the catalogue on typed questions (Choice, Score, Noul), measured on Typed Decisions.

The catalogue is trained on the training split with `KV.train_questions` and answers the test split alone
(`KV.decide(..., head=...)`, no LLM call). Targets: the official most probable answer (hard), the official
distribution (soft), or the answers of an LLM teacher (`KV.teach`, no human label at all). The untrained catalogue
is the reference.

    python benchmarks/typed_train_bench.py --teachers qwen3:8b,hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ2_S
"""
import argparse
import json
import os
import statistics
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DATA, gold_key, load_json, save_json, to_text, typed_decisions  # noqa: E402
from kvindex import KV  # noqa: E402
from kvindex.train import Head  # noqa: E402


def predict_key(q, answer):
    """The answer as a key comparable with `gold_key`."""
    if q["type"] == "choice":
        return answer.choice
    elif q["type"] == "score":
        return max(answer.probabilities, key=answer.probabilities.get)
    else:  # noul
        return "true" if answer.noul >= 0.5 else "false"


def evaluate(kv, head, cases):
    """Accuracy over the gold decisions, accuracy by type, median ms per case (one decide call per case) and per
    question."""
    hits, by_type, ms = [], {}, []
    for case in cases:
        state = to_text(case["state"])
        t0 = time.perf_counter()
        r = kv.decide(state, case["questions"], head=head)
        ms.append((time.perf_counter() - t0) * 1000)
        for qid, g in case["gold"].items():
            q = case["questions"][qid]
            h = predict_key(q, r.answers[qid]) == gold_key(g)
            hits.append(h)
            by_type.setdefault(q["type"], []).append(h)
    ms_per_question = statistics.median(ms) * len(cases) / sum(len(c["questions"]) for c in cases)
    return {"acc": float(np.mean(hits)), "ms": ms_per_question, "ms_per_case": statistics.median(ms),
            "decisions": len(hits), "by_type": {t: float(np.mean(v)) for t, v in by_type.items()}}


def gold_examples(cases, soft):
    """(state, question, answer) for every gold decision: the most probable key (hard) or the distribution (soft)."""
    examples = []
    for case in cases:
        for qid, g in case["gold"].items():
            answer = g["probabilities"] if soft else gold_key(g)
            examples.append((to_text(case["state"]), case["questions"][qid], answer))
    return examples


def teacher_file(llm):
    """Path of the saved teacher labels for one LLM."""
    return os.path.join(DATA, "teach", llm.replace("hf.co/", "").replace(":", "_").replace("/", "_") + ".json")


def teach(llm, cases, a):
    """Teacher labels for the training cases (computed once and saved, as plain distributions)."""
    path = teacher_file(llm)
    if os.path.exists(path):
        return json.load(open(path, encoding="utf8"))
    kv = KV(llm, n_ctx=a.n_ctx, kv_type=a.kv_type)
    try:
        t0 = time.perf_counter()
        ex = kv.teach([(to_text(c["state"]), c["questions"]) for c in cases])
        seconds = time.perf_counter() - t0
    finally:
        kv.close()
    labels = []
    for (state, q, ans) in ex:
        if q["type"] in ("choice", "score"):
            dist = ans.probabilities
        else:
            dist = {"true": ans.noul, "false": 1 - ans.noul}
        labels.append({"state": state, "question": q, "answer": dist})
    out = {"seconds": seconds, "labels": labels}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w", encoding="utf8"))
    return out


def main():
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="qwen3:8b")
    parser.add_argument("--embed", default="qwen3-embedding:4b")
    parser.add_argument("--teachers", default="")
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--kv-type", default="q8_0")
    parser.add_argument("--limit", type=int, default=0)
    a = parser.parse_args()
    out_name = "typed_train_qwen3_8b.json"
    res = load_json(out_name)
    train = typed_decisions("train")
    test = typed_decisions("test")
    if a.limit:
        train = train[:a.limit]
    teachers = [t for t in a.teachers.split(",") if t]
    taught = {t: teach(t, train, a) for t in teachers}
    kv = KV(a.llm, embed=a.embed, n_ctx=a.n_ctx, kv_type=a.kv_type)
    try:
        d = kv.encoder(["x"]).shape[1]
        z = np.zeros((d, 1), dtype=np.float32)

        def log(name, row):
            res[f"typed/{name}"] = row
            save_json(out_name, res)
            print(f"{name:<28} {row}", flush=True)

        log("catalogue", evaluate(kv, Head(z, z, z, z, {"scale": 30.0}), test))
        for name, soft in (("trained-hard", False), ("trained-soft", True)):
            t0 = time.perf_counter()
            head = kv.train_questions(gold_examples(train, soft))
            row = evaluate(kv, head, test)
            row["train"] = dict(head.report, total_s=time.perf_counter() - t0)
            log(name, row)
        for t, data in taught.items():
            ex = [(x["state"], x["question"], x["answer"]) for x in data["labels"]]
            t0 = time.perf_counter()
            head = kv.train_questions(ex)
            row = evaluate(kv, head, test)
            row["train"] = dict(head.report, total_s=time.perf_counter() - t0, teacher_s=data["seconds"])
            log("trained-teacher-" + t.replace("hf.co/", "").replace(":", "_").replace("/", "_"), row)
    finally:
        kv.close()


if __name__ == "__main__":
    main()
