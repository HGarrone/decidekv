"""Same tail, by hand: in sequence 0 after copying the prefix (reference path) and in a spare sequence."""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, r"E:\AI\PAI_1shot_opt")
sys.path.insert(0, r"E:\AI\PAI_1shot_opt\benchmarks")
from decidekv import KV  # noqa: E402
from data import to_text, typed_decisions  # noqa: E402
from verify_resident import QS  # noqa: E402


def last_probs(be, rows=1, row=0):
    x = np.ctypeslib.as_array(be.L.llama_get_logits(be.ctx), shape=(rows, be.n_vocab))[row].astype(np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    warnings.simplefilter("ignore")
    kv = KV(sys.argv[1], n_ctx=8192, kv_type=sys.argv[2] if len(sys.argv) > 2 else "q8_0")
    be = kv.client
    text = " ".join(to_text(c["state"]) for c in typed_decisions("test")[:9])
    reqs = []
    for qid, q in QS.items():
        q, idx, ins = kv.question_index(q, warm=False)
        idx.warm(ins)
        be.hold(idx.segments("", ins, "options-first")[0])
        reqs.append((qid, idx.label_request(text, ins, idx.keys, "options-first")))
    print("distinct first segments:", len({r[1][0][0] for r in reqs}), "bases:", list(be.bases))
    for qid, (segs, labels) in reqs[:6]:
        toks, n = be._split(segs)
        p0, tail = toks[:n], toks[n:]
        base, _ = be._base(p0)
        be.seq_rm(0, -1, -1)
        be.seq_cp(base, 0)
        be._decode(tail, n, 0)
        a = last_probs(be)
        spare = max(be.resident + be.scratch, 9)
        be.seq_rm(spare, -1, -1)
        be.seq_cp(base, spare)
        be._decode(tail, n, spare)
        b = last_probs(be)
        be.seq_rm(spare, -1, -1)
        be.seq_rm(0, -1, -1)
        lp_ref, _ = be.label_logprobs(segs, labels)
        c = be.label_logprobs_resident([(segs, labels)])[0][0]
        ids = {lab: {be.tokenize(v)[0] for v in (lab, " " + lab)} for lab in labels}
        top = {lab: round(float(sum(a[i] for i in ids[lab])), 3) for lab in labels}
        print(qid, "base", base, "p0", len(p0), "tail", len(tail), "| seq0 vs spare: max|dp| =",
              round(float(np.abs(a - b).max()), 4), "| ref", {k: round(float(np.exp(v)), 3) for k, v in lp_ref.items()},
              "| resident", {k: round(float(np.exp(v)), 3) for k, v in c.items()}, "manual", top, flush=True)
    kv.close()


if __name__ == "__main__":
    main()
