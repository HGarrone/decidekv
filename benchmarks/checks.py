"""
Correctness checks that must pass before any benchmark is run.

    python benchmarks/checks.py --llm qwen3:8b

1. chat template: thinking disabled (Qwen3-style templates)
2. prefix reuse: logits after reusing the cached prefix == logits after a full recomputation
3. resident prefix (forked into the working sequence) == decoding without cache
4. resident prefixes do not leak into each other
5. eviction and re-creation of resident prefixes

With --baselines, the reference methods of benchmarks/baselines.py are also run on three trivial cases.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decidekv.llamacpp import LlamaCpp  # noqa: E402

INTRO = ("You are a decision engine. Answer the question about the definition by choosing one word from the "
         "list of words. Reply with the word exactly as written in the list, and nothing else.\n\n### Words\n")
OPTIONS = ["- river\n", "- mountain\n", "- question\n", "- attention\n"]
TAIL = "\n### Definition\na natural stream of water flowing to the sea\n\n### Question\nWhich word does this definition define?\n\n### Word\n"

TAIL2 = TAIL.replace("a natural stream of water flowing to the sea", "the act of asking for information")
LIST_B = ["- doctor\n", "- pilot\n", "- farmer\n", "- singer\n", "- lawyer\n"]
LIST_C = ["- red\n", "- green\n", "- blue\n", "- yellow\n", "- purple\n", "- orange\n"]


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def logits(be):
    return be.logprobs_last().copy()


def close(a, b, tol):
    """a, b: log-probabilities over the vocabulary. Compared as probabilities: max |p_a - p_b| and KL(a||b)."""
    pa, pb = np.exp(a), np.exp(b)
    dp = float(np.max(np.abs(pa - pb)))
    kl = float(np.sum(pa * (a - b)))
    same_top = int(np.argmax(a)) == int(np.argmax(b))
    return dp <= tol and same_top, f"(max |dp| = {dp:.2e}, KL = {kl:.2e}, same argmax: {same_top})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="qwen3:8b")
    ap.add_argument("--kv-type", default="f16")
    ap.add_argument("--baselines", action="store_true", help="also check the reference methods on trivial cases")
    a = ap.parse_args()
    be = LlamaCpp(a.llm, n_ctx=4096, kv_type=a.kv_type, resident=2)
    ok = True

    # 1. template
    wrap_after = be._wrap[1]
    thinking_off = "<think>" not in wrap_after or "</think>" in wrap_after
    ok &= report("chat template, thinking disabled", thinking_off, repr(wrap_after[-40:]))

    pre = be.bos() + be.tokenize(be._wrap[0] + INTRO, special=True)
    opt = [be.tokenize(o) for o in OPTIONS]
    tail = be.tokenize(TAIL + be._wrap[1], special=True)
    full = pre + sum(opt, []) + tail

    # 2. prefix reuse vs full recomputation
    be.clear()
    be.eval_tokens(pre + sum(opt, []))
    be.eval_tokens(full)                        # reuses the prefix, evaluates the tail only
    cached = logits(be)
    be.clear()
    be.eval_tokens(full)
    fresh = logits(be)
    ok &= report("prefix reuse == full recomputation", *close(cached, fresh, 1e-3))

    # 3-5. resident prefixes (every architecture): exact, isolated, evicted and re-created
    seg_a = (INTRO + "".join(OPTIONS), TAIL)
    seg_a2 = (INTRO + "".join(OPTIONS), TAIL2)
    seg_b = (INTRO + "".join(LIST_B), TAIL)
    seg_c = (INTRO + "".join(LIST_C), TAIL)

    toks, n = be._split(seg_a2)
    be.clear()
    be._decode(toks[:n], 0, 0)
    be._decode(toks[n:], n, 0)
    ref = logits(be)
    be.clear()
    be._registered.clear()
    be.prefill(seg_a)
    stats = be._eval(seg_a2)
    got = logits(be)
    same, detail = close(got, ref, 1e-6)
    ok &= report("resident prefix == decoding without cache", same and stats["cached_tokens"] == n,
                 f"{detail} reused {stats['cached_tokens']} of {n} prefix tokens")

    other = be.tokenize("".join(LIST_B) * 3)
    other2 = be.tokenize("".join(LIST_C) * 3)[:len(other)]
    runs = []
    for junk in (other, other2):
        be.clear()
        be._registered.clear()
        be._base(be.bos() + junk)
        be.prefill(seg_a)
        be._eval(seg_a2)
        runs.append(logits(be))
    leak = float(np.max(np.abs(np.exp(runs[0]) - np.exp(runs[1]))))
    drift = float(np.max(np.abs(np.exp(runs[0]) - np.exp(ref))))
    ok &= report("resident prefixes do not leak into each other", len(other2) == len(other) and leak == 0.0,
                 f"(max |dp| between two different neighbours = {leak:.2e}; numerical drift against the prefix alone "
                 f"in the cache = {drift:.2e}, information)")

    be.clear()
    be._registered.clear()
    be.prefill(seg_a)
    be.prefill(seg_b)
    be.prefill(seg_c)
    kept = len(be.bases)
    first = be._eval(seg_a2)
    top1 = int(np.argmax(logits(be)))
    second = be._eval(seg_a2)
    top2 = int(np.argmax(logits(be)))
    good = (kept == 2 and first["cached_tokens"] == 0 and second["cached_tokens"] == n
            and top1 == int(np.argmax(ref)) and top2 == top1)
    ok &= report("eviction and re-creation of resident prefixes", good,
                 f"(resident {kept}, reused tokens after eviction {first['cached_tokens']}, then {second['cached_tokens']})")

    be.close()
    if a.baselines:
        ok &= baselines_check(a)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


def baselines_check(a):
    """Every reference method must solve trivial cases (catches wiring errors: wrong readout, order, heads)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import clm
    from baselines import QUERY_TEMPLATE, openjev_pick, run_cross
    from decidekv import KV
    from decidekv.core import Index
    from decidekv.llamacpp import LlamaCppEncoder
    words = ["river", "mountain", "question", "attention", "bicycle", "kitchen"]
    cases = [("a natural stream of water flowing to the sea", {"river"}),
             ("a room where food is prepared and cooked", {"kitchen"}),
             ("a vehicle with two wheels moved by pedals", {"bicycle"})]
    prompt = dict(noun="word", state_label="Definition", question="Which word does this definition define?")
    ok = True
    kv = KV(a.llm, backend="llamacpp", n_ctx=4096, kv_type=a.kv_type)
    be = kv.client
    ix = Index(kv, words, warm=False, **prompt)
    got = [openjev_pick(be, ix, s) for s, _ in cases]
    ok &= report("openjev readout (state-first, first token) solves trivial cases",
                 all(g in gold for g, (_, gold) in zip(got, cases)), str(got))
    got = [ix._match(be.pick_option(ix.segments(s, None, order="state-first"), ix.keys)[0] or "") for s, _ in cases]
    ok &= report("state-first decoding solves trivial cases", all(g in gold for g, (_, gold) in zip(got, cases)), str(got))
    row = run_cross(kv, be, words, cases, prompt)
    ok &= report("cross-encoder (yes/no per option) solves trivial cases", row["acc"] == 1.0, str(row))
    be.close()
    enc = LlamaCppEncoder("qwen3-embedding:4b")
    D = enc(words)
    got = [words[int(np.argmax(D @ enc([QUERY_TEMPLATE.format(question=prompt["question"], state=s)])[0]))]
           for s, _ in cases]
    ok &= report("embedding baseline solves trivial cases", all(g in gold for g, (_, gold) in zip(got, cases)), str(got))
    enc.close()
    # CLM-8B on the example of its README: the released heads must rank the right answer first
    enc = LlamaCppEncoder(a.llm, pooling="last")
    heads = clm.Heads()
    cands = ["The Moon's gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."]
    zs = heads.project(enc([clm.state_text("", "What causes tides on Earth?")]), "state")[0]
    s = heads.project(enc(cands), "action") @ zs
    ok &= report("CLM-8B (encoder + released heads) ranks the README example correctly", int(np.argmax(s)) == 0,
                 str(np.round(s, 3).tolist()))
    zs = [heads.project(enc([clm.state_text(f"Definition: {d}", prompt["question"])]), "state")[0] for d, _ in cases]
    za = heads.project(enc(words), "action")
    got = [words[int(np.argmax(za @ z))] for z in zs]
    report("CLM-8B zero-shot on the trivial definitions (information)", True, str(got))
    enc.close()
    return ok


if __name__ == "__main__":
    main()
