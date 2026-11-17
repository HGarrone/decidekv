"""
Cascade on Typed Decisions: the trained catalogue answers every question; the questions whose answer has a confidence
below a threshold go to the LLM (`KV.decide(..., head=..., threshold=...)`).

The 27B model and the embedding model do not fit together in 12 GB, so the cascade runs in two phases with the same
arithmetic as `decide`: phase "catalogue" (embedding model only) trains the heads and stores the catalogue's answers,
confidences and time per case; phase "llm" (LLM only) asks the LLM the questions below each threshold, case by case,
in one `decide` call, and adds the two times.

    python benchmarks/cascade_bench.py --phase catalogue
    python benchmarks/cascade_bench.py --phase llm --llm qwen3:8b
    python benchmarks/cascade_bench.py --phase llm --llm hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ2_S --n-ctx 8192
"""
import argparse
import json
import os
import statistics
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DATA, gold_key, load_json, save_json, to_text, typed_decisions  # noqa: E402
from decidekv import KV  # noqa: E402
from decidekv.llamacpp import LlamaCppEncoder  # noqa: E402
from decidekv.primitives import answer_confidence  # noqa: E402
from typed_train_bench import gold_examples, predict_key, teacher_file  # noqa: E402

THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
TEACHER = "hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ2_S"


def cascade_file(head_name):
    """Path of the stored catalogue answers of one head."""
    return os.path.join(DATA, "cascade", head_name + ".json")


def catalogue_phase(a):
    """Train the two heads, answer the test split with each, store answers, confidences and times."""
    train = typed_decisions("train")
    test = typed_decisions("test")
    heads = {"human": gold_examples(train, True)}
    if os.path.exists(teacher_file(TEACHER)):
        with open(teacher_file(TEACHER), encoding="utf8") as f:
            data = json.load(f)
        heads["teacher-27b"] = [(x["state"], x["question"], x["answer"]) for x in data["labels"]]
    enc = LlamaCppEncoder(a.embed)
    kv = KV("catalogue-only", backend="ollama", embed=enc)
    try:
        for name, examples in heads.items():
            head = kv.train_questions(examples)
            rows = []
            ms_list = []
            for case in test:
                state = to_text(case["state"])
                start = time.perf_counter()
                r = kv.decide(state, case["questions"], head=head)
                end = time.perf_counter()
                ms = (end - start) * 1000
                ms_list.append(ms)
                rows.append({
                    "ms": ms,
                    "answers": {
                        qid: {
                            "key": predict_key(q, r.answers[qid]),
                            "conf": answer_confidence(r.answers[qid])
                        }
                        for qid, q in case["questions"].items()
                    }
                })
            os.makedirs(os.path.dirname(cascade_file(name)), exist_ok=True)
            with open(cascade_file(name), "w", encoding="utf8") as f:
                json.dump({"report": head.report, "cases": rows}, f)
            hits = []
            for case, row in zip(test, rows):
                for qid, g in case["gold"].items():
                    hits.append(row["answers"][qid]["key"] == gold_key(g))
            acc = sum(hits) / len(hits)
            print(f"{name}: catalogue alone {acc:.3f}, median {statistics.median(ms_list):.0f} ms per case")
    finally:
        enc.close()


def llm_phase(a):
    """For every stored head and threshold: the questions below the threshold go to the LLM, case by case."""
    test = typed_decisions("test")
    label = a.llm.replace("hf.co/", "").replace(":", "_").replace("/", "_")
    out_name = f"cascade_{label}.json"
    res = load_json(out_name)
    kv = KV(a.llm, n_ctx=a.n_ctx, kv_type=a.kv_type)
    try:
        for name in ("human", "teacher-27b"):
            if not os.path.exists(cascade_file(name)):
                continue
            with open(cascade_file(name), encoding="utf8") as f:
                stored = json.load(f)
            for t in THRESHOLDS:
                hits, ms, sent, total = [], [], 0, 0
                for case, row in zip(test, stored["cases"]):
                    state = to_text(case["state"])
                    qs = case["questions"]
                    keys = {qid: v["key"] for qid, v in row["answers"].items()}
                    low = [qid for qid, v in row["answers"].items() if v["conf"] < t]
                    llm_ms = 0.0
                    if low:
                        start = time.perf_counter()
                        r = kv.decide(state, {qid: qs[qid] for qid in low})
                        end = time.perf_counter()
                        llm_ms = (end - start) * 1000
                        for qid in low:
                            keys[qid] = predict_key(qs[qid], r.answers[qid])
                    ms.append((row["ms"] + llm_ms) / len(qs))
                    sent += len(low)
                    total += len(qs)
                    for qid, g in case["gold"].items():
                        hits.append(keys[qid] == gold_key(g))
                res[f"{name}/T={t}"] = {
                    "acc": sum(hits) / len(hits),
                    "ms": statistics.median(ms),
                    "escalated": sent / total,
                    "decisions": len(hits)
                }
                save_json(out_name, res)
                print(f"{name}/T={t}: {res[f'{name}/T={t}']}", flush=True)
    finally:
        kv.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["catalogue", "llm"])
    parser.add_argument("--embed", default="qwen3-embedding:4b")
    parser.add_argument("--llm", default="qwen3:8b")
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--kv-type", default="q8_0")
    a = parser.parse_args()
    warnings.simplefilter("ignore")
    if a.phase == "catalogue":
        catalogue_phase(a)
    else:
        llm_phase(a)


if __name__ == "__main__":
    main()
