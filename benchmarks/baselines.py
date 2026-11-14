"""
kvindex and the reference methods under equal conditions (growing lists of kvindex alone: benchmarks/scale.py).

The same engine (llama.cpp, in process), the same GGUF model (Qwen3-8B, the backbone of CLM-8B), the same KV-cache
type, the same queries and the same prompt blocks (kvindex.core.Index: instructions, options, state, question, answer
head). LLM methods differ only in the order of the blocks, in what stays cached and in how the answer is read.

  kvindex     options first, kept in the KV cache; the answer is decoded by name through the option trie
              (the cached list, up to 3,000 options; larger lists are measured by benchmarks/scale.py)
  state-first the same blocks and decoding with the options after the state (nothing reusable across decisions)
  openjev     OpenJev's architecture: options after the state, one forward pass, first-token logit of every option
              (at most 255 options, OpenJev's limit)
  cross       one yes/no question per option (cross-encoder reference; state cached; up to 300 options, 30 queries)
  (state-first runs up to 10,000 options: beyond, the whole list would be recomputed at every decision)
  emb         generic embedding model alone (qwen3-embedding:4b), cosine between query and options, no training
  emb+llm     top-50 of the embedding model, then the LLM chooses among them (retrieve and rerank, nothing cached)
  clm8b       the authors' CLM-8B: Qwen3-8B last-token embedding + released heads, zero-shot (benchmarks/clm.py)
  clm8b-ft    CLM-8B with the heads trained further on the task's training split (best validation epoch)
  prior       most frequent training answer per workflow and question (typed only; ignores the input)
  random      a random cell among those not refuted yet (aim only)

Tasks
  wordnet     reverse dictionary (definition -> word), 150 definitions, 200 to 30,000 options
  banking77   77 intents, 300 customer messages
  typed       Typed Decisions test split (400 cases, five typed questions each, official scoring)
  aim         a hidden target on a 15x15 grid (225 cells); after a miss only the direction is told; 20 games,
              at most 25 shots; refuted cells are excluded

    python benchmarks/baselines.py --task wordnet --sizes 200,1000,3000
    python benchmarks/baselines.py --task banking77
    python benchmarks/baselines.py --task typed
    python benchmarks/baselines.py --task aim
Results: results/baselines_<llm>.json, keys "<task>/<size>/<method>".
"""
import argparse
import gc
import hashlib
import math
import os
import random
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import clm  # noqa: E402
from data import (DATA, banking77, banking77_train, correct, gold_key, load_json, save_json, to_text,  # noqa: E402
                  typed_decisions, wordnet, wordnet_train)
from kvindex import KV  # noqa: E402
from kvindex.core import Index, render_state  # noqa: E402
from kvindex.catalog import QUERY_TEMPLATE  # noqa: E402
from kvindex.llamacpp import LlamaCppEncoder  # noqa: E402

LIST_PROMPTS = {
    "wordnet": dict(noun="word", state_label="Definition", question="Which word does this definition define?"),
    "banking77": dict(noun="intent", state_label="Customer message",
                      question="Which intent does this customer message express?"),
}
METHODS = {
    "wordnet": ["emb", "emb+llm", "kvindex", "state-first", "openjev", "cross", "clm8b", "clm8b-ft"],
    "banking77": ["emb", "emb+llm", "kvindex", "state-first", "openjev", "cross", "clm8b", "clm8b-ft"],
    "typed": ["prior", "emb", "kvindex", "openjev", "clm8b", "clm8b-ft"],
    "aim": ["random", "kvindex", "openjev", "clm8b"],
}
TOP_K = 50
CROSS_QUERIES = 30
LIMITS = {   # method -> (largest option set, reason); beyond it the method is recorded as skipped
    "openjev": (255, "not supported: OpenJev accepts at most 255 options"),
    "cross": (300, "skipped: one LLM call per option"),
    "state-first": (10000, "skipped: the whole option list is recomputed for every decision (does not fit in memory)"),
}


# ---------------------------------------------------------------- shared helpers
def cached_vectors(enc, texts, tag):
    """Embeddings of `texts` (offline part), cached on disk by content."""
    h = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
    path = os.path.join(DATA, "cache", f"{tag}_{h}.npy")
    if os.path.exists(path):
        return np.load(path)
    v = enc(list(texts))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, v)
    return v


def flat_ctx(n_tokens_estimate):
    return int(2 ** math.ceil(math.log2(n_tokens_estimate + 1024)))


def timed(queries, decide, extra_ms=None):
    """Accuracy and median time per decision; `extra_ms[i]` is added to decision i (work done in another phase)."""
    ok, ms = 0, []
    for i, (state, gold) in enumerate(queries):
        t0 = time.perf_counter()
        answer = decide(state)
        ms.append(1000 * (time.perf_counter() - t0) + (extra_ms[i] if extra_ms else 0.0))
        ok += correct(answer, gold)
    return {"acc": ok / len(queries), "ms": statistics.median(ms), "queries": len(queries)}


def first_max(keys, scores):
    """argmax with ties broken by list order (as OpenJev does)."""
    return keys[max(range(len(keys)), key=lambda i: scores[keys[i]])]


def openjev_pick(be, ix, state, instructions=None, labels=None):
    """OpenJev's readout on the shared blocks: state-first order, logit of the first token of each label."""
    labels = labels or ix.keys
    lp, _ = be.first_token_logprobs(ix.segments(state, instructions, order="state-first"), labels)
    return first_max(labels, lp)


class Runner:
    def __init__(self, a):
        self.a = a
        tag = a.llm.replace("hf.co/", "").replace(":", "_").replace("/", "_")
        self.out_name = f"baselines_{tag}.json"
        self.res = load_json(self.out_name)

    def want(self, task, n, method):
        return method in self.a.methods and (self.a.redo or f"{task}/{n}/{method}" not in self.res)

    def log(self, task, n, method, row):
        self.res[f"{task}/{n}/{method}"] = row
        save_json(self.out_name, self.res)
        show = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
        print(f"{task:>9} {n:>6} {method:<12} {show}", flush=True)

    def kv(self, n_ctx):
        return KV(self.a.llm, backend="llamacpp", n_ctx=n_ctx, kv_type=self.a.kv_type)


# ---------------------------------------------------------------- list tasks: wordnet, banking77
def list_task(r: Runner, task: str, n: int):
    a = r.a
    if task == "wordnet":
        options, queries = wordnet(n, a.queries)
    else:
        options, queries = banking77(a.queries)
        n = len(options)
    prompt = LIST_PROMPTS[task]
    todo = [m for m in METHODS[task] if r.want(task, n, m)]
    for m in [m for m in todo if m in LIMITS and n > LIMITS[m][0]]:
        r.log(task, n, m, {"skipped": LIMITS[m][1]})
        todo.remove(m)
    if not todo:
        return
    # 1. embedding model: emb, and the shortlists (and their cost) for emb+llm
    tops, emb_ms = None, None
    if "emb" in todo or "emb+llm" in todo:
        enc = LlamaCppEncoder(a.embed)
        vec = cached_vectors(enc, options, "emb_" + a.embed.replace(":", "_"))
        enc(["warm-up"])
        tops, emb_ms, ok = [], [], 0
        for state, gold in queries:
            t0 = time.perf_counter()
            v = enc([QUERY_TEMPLATE.format(question=prompt["question"], state=render_state(state))])[0]
            s = vec @ v
            k = min(TOP_K, n)
            top = np.argpartition(-s, k - 1)[:k]
            top = top[np.argsort(-s[top])]
            emb_ms.append(1000 * (time.perf_counter() - t0))
            tops.append([int(i) for i in top])
            ok += correct(options[top[0]], gold)
        if "emb" in todo:
            r.log(task, n, "emb", {"acc": ok / len(queries), "ms": statistics.median(emb_ms), "queries": len(queries),
                                   "recall_top_k": sum(any(correct(options[i], g) for i in t)
                                                       for t, (_, g) in zip(tops, queries)) / len(queries)})
        enc.close()
        del enc
        gc.collect()
    # 2. LLM methods
    llm_methods = [m for m in todo if m in ("kvindex", "state-first", "openjev", "emb+llm", "cross")]
    if llm_methods:
        try:
            kv = r.kv(flat_ctx(4 * n))
        except Exception as e:                       # the whole option list does not fit in memory
            for m in llm_methods:
                if m != "emb+llm":
                    r.log(task, n, m, {"error": f"{type(e).__name__}: {e}"})
            kv = r.kv(16384) if "emb+llm" in llm_methods else None
        if kv is not None:
            be = kv.client
            ix = Index(kv, options, warm=False, **prompt)
            llm_list_methods(r, task, n, kv, be, ix, options, queries, prompt, tops, emb_ms, llm_methods)
            be.close()
            del kv, be
            gc.collect()
    # 3. CLM-8B
    if "clm8b" in todo or "clm8b-ft" in todo:
        if task == "wordnet":
            pairs, split_by_action = wordnet_train({d for d, _ in queries}), True
        else:
            pairs, split_by_action = banking77_train(), False
        clm_list(r, task, n, options, queries, prompt, pairs, split_by_action, todo)


def llm_list_methods(r, task, n, kv, be, ix, options, queries, prompt, tops, emb_ms, methods):
    def safe(method, fn):
        try:
            r.log(task, n, method, fn())
        except Exception as e:
            r.log(task, n, method, {"error": f"{type(e).__name__}: {e}"})

    if "kvindex" in methods:
        def run_kvindex():
            t0 = time.perf_counter()
            idx = kv.index(options, **prompt)
            build = time.perf_counter() - t0
            row = timed(queries, lambda s: idx.choose(s).choice)
            row["build_s"] = build
            return row
        safe("kvindex", run_kvindex)
    if "state-first" in methods:
        safe("state-first", lambda: timed(queries, lambda s: ix._match(
            be.pick_option(ix.segments(s, None, order="state-first"), ix.keys)[0] or "")))
    if "openjev" in methods:
        safe("openjev", lambda: timed(queries, lambda s: openjev_pick(be, ix, s)))
    if "emb+llm" in methods:
        it = iter(tops)
        safe("emb+llm", lambda: timed(queries, lambda s: Index(kv, [options[i] for i in next(it)], warm=False,
                                                               **prompt).choose(s).choice, extra_ms=emb_ms))
    if "cross" in methods:
        safe("cross", lambda: run_cross(kv, be, options, queries[:CROSS_QUERIES], prompt))


def run_cross(kv, be, options, queries, prompt):
    """One yes/no question per option; [instructions + yes/no block] and [state] are reused from the cache, only the
    question about each option is evaluated."""
    yn = kv.index({"Yes": "the answer to the question is yes / true", "No": "the answer to the question is no / false"},
                  labels=["Yes", "No"], noun="answer", state_label=prompt["state_label"])
    noun = prompt["noun"]

    def decide(state):
        best, best_p = options[0], -1.0
        for o in options:
            lps, _ = be.label_logprobs(yn.segments(state, f'{prompt["question"]} Is the {noun} "{o}"?'), ["Yes", "No"])
            p = 1.0 / (1.0 + math.exp(lps["No"] - lps["Yes"]))
            if p > best_p:
                best, best_p = o, p
        return best
    row = timed(queries, decide)
    row["llm_calls_per_decision"] = len(options)
    return row


def clm_list(r, task, n, options, queries, prompt, pairs, split_by_action, todo):
    a = r.a
    enc = LlamaCppEncoder(a.llm, pooling="last")
    tag = "clm_" + a.llm.replace(":", "_")
    heads = clm.Heads()
    E_opt = cached_vectors(enc, options, tag)
    label, question = prompt["state_label"], prompt["question"]
    texts = [clm.state_text(f"{label}: {s}", question) for s, _ in queries]
    enc(["warm-up"])
    E_q, enc_ms = [], []
    for t in texts:                                           # online part: one state at a time
        t0 = time.perf_counter()
        E_q.append(enc([t])[0])
        enc_ms.append(1000 * (time.perf_counter() - t0))
    E_q = np.stack(E_q)

    def score(h, name, extra=None):
        za = h.project(E_opt, "action")
        it = iter(range(len(queries)))

        def decide(_state):
            i = next(it)
            zs = h.project(E_q[i:i + 1], "state")[0]
            return options[int(np.argmax(za @ zs))]
        row = timed(queries, decide, extra_ms=enc_ms)
        row.update(extra or {})
        r.log(task, n, name, row)

    if "clm8b" in todo:
        score(heads, "clm8b")
    if "clm8b-ft" in todo:
        ft_path = os.path.join(DATA, "clm", f"heads_ft_{task}.npz")
        info_key = f"{task}/train/clm8b-ft"
        if not os.path.exists(ft_path) or info_key not in r.res:
            actions = list(dict.fromkeys(x for _, x in pairs))
            amap = {x: i for i, x in enumerate(actions)}
            S = cached_vectors(enc, [clm.state_text(f"{label}: {s}", question) for s, _ in pairs], tag)
            A = cached_vectors(enc, actions, tag)
            print(f"    training CLM-8B heads on {len(pairs)} {task} pairs", flush=True)
            info = clm.finetune(dict(S=S, A=A, ids=np.array([amap[x] for _, x in pairs]),
                                     split_by_action=np.array(split_by_action)), ft_path)
            r.res[info_key] = info
            save_json(r.out_name, r.res)
        score(clm.Heads(ft_path), "clm8b-ft", {"train": r.res[info_key]})
    enc.close()
    del enc
    gc.collect()


# ---------------------------------------------------------------- typed decisions
NOUL_KEYS = ("false", "true")


def clm_candidates(q):
    """Option keys and texts in CLM's own format."""
    t, crit, ins = q["type"], q.get("criteria"), to_text(q.get("instructions")).strip()
    if t == "choice":
        keys = list(crit)
        return keys, [to_text(crit[k]) if crit[k] not in (None, "") else k for k in keys]
    if t == "score":
        return [str(i) for i in range(len(crit))], [to_text(c) for c in crit]
    crit = crit if isinstance(crit, dict) else {}
    texts = []
    for k in NOUL_KEYS:
        d = crit.get(k)
        if d in (None, ""):
            d = (f"Yes. This is true: {ins}" if k == "true" else f"No. This is false: {ins}") if ins else k
        texts.append(f"{k}: {to_text(d)}")
    return list(NOUL_KEYS), texts


def typed_task(r: Runner):
    a = r.a
    test = typed_decisions("test")
    if a.limit:
        test = test[:a.limit]
    todo = [m for m in METHODS["typed"] if r.want("typed", "all", m)]
    if not todo:
        return
    n_dec = sum(len(c["gold"]) for c in test)

    def report(name, preds, ms, extra=None):
        hits, by_type = [], {}
        for c, p in zip(test, preds):
            for qid, g in c["gold"].items():
                h = p.get(qid) == gold_key(g)
                hits.append(h)
                by_type.setdefault(g["type"], []).append(h)
        row = {"acc": float(np.mean(hits)), "ms": statistics.median(ms) if ms else 0.0, "decisions": n_dec,
               "by_type": {t: float(np.mean(v)) for t, v in by_type.items()}}
        row.update(extra or {})
        r.log("typed", "all", name, row)

    if "prior" in todo:
        counts = {}
        for c in typed_decisions("train"):
            for qid, g in c["gold"].items():
                d = counts.setdefault((c["workflow"], qid), {})
                d[gold_key(g)] = d.get(gold_key(g), 0) + 1
        report("prior", [{qid: max(counts.get((c["workflow"], qid), {"": 0}).items(), key=lambda t: t[1])[0]
                          for qid in c["gold"]} for c in test], [])
    if "emb" in todo:
        enc = LlamaCppEncoder(a.embed)
        tag = "emb_" + a.embed.replace(":", "_")
        cand_texts = sorted({t for c in test for q in c["questions"].values() for t in clm_candidates(q)[1]})
        C = dict(zip(cand_texts, cached_vectors(enc, cand_texts, tag)))
        enc(["warm-up"])
        preds, ms = [], []
        for c in test:
            qs = list(c["questions"].items())
            t0 = time.perf_counter()
            V = enc([QUERY_TEMPLATE.format(question=to_text(q.get("instructions")), state=to_text(c["state"]))
                     for _, q in qs])
            per = 1000 * (time.perf_counter() - t0) / len(qs)
            p = {}
            for (qid, q), v in zip(qs, V):
                keys, texts = clm_candidates(q)
                p[qid] = keys[int(np.argmax([C[t] @ v for t in texts]))]
            preds.append(p)
            ms += [per] * len(qs)
        report("emb", preds, ms)
        enc.close()
        del enc
        gc.collect()
    if "kvindex" in todo or "openjev" in todo:
        kv = r.kv(a.n_ctx)
        be = kv.client
        for method in ("kvindex", "openjev"):
            if method not in todo:
                continue
            preds, ms = [], []
            for c in test:
                state = to_text(c["state"])
                p = {}
                for qid, qd in c["questions"].items():
                    t0 = time.perf_counter()
                    if method == "kvindex":
                        ans = kv.decide(state, {qid: qd}).answers[qid]
                        if ans.type == "choice":
                            p[qid] = ans.choice
                        elif ans.type == "score":
                            p[qid] = max(ans.probabilities, key=ans.probabilities.get)
                        else:
                            p[qid] = "true" if ans.noul >= 0.5 else "false"
                    else:
                        q, ix, instr = kv.question_index(qd, warm=False)
                        lab = openjev_pick(be, ix, state, instr, ix.labels or ix.keys)
                        key = ix.keys[(ix.labels or ix.keys).index(lab)]
                        p[qid] = key if q.type != "noul" else ("true" if key == "Yes" else "false")
                    ms.append(1000 * (time.perf_counter() - t0))
                preds.append(p)
            report(method, preds, ms)
        be.close()
        del kv, be
        gc.collect()
    if "clm8b" in todo or "clm8b-ft" in todo:
        typed_clm(r, test, todo, report)


def typed_clm(r, test, todo, report):
    a = r.a
    train = typed_decisions("train")
    enc = LlamaCppEncoder(a.llm, pooling="last")
    tag = "clm_" + a.llm.replace(":", "_")

    def items(cases):
        out = []
        for ci, c in enumerate(cases):
            for qid, q in c["questions"].items():
                keys, texts = clm_candidates(q)
                g = keys.index(gold_key(c["gold"][qid])) if qid in c["gold"] else -1
                out.append((ci, qid, clm.state_text(to_text(c["state"]), to_text(q.get("instructions"))), keys, texts, g))
        return out
    tr, te = items(train), items(test)
    cand_texts = sorted({t for it in tr + te for t in it[4]})
    C = cached_vectors(enc, cand_texts, tag)
    cid = {t: i for i, t in enumerate(cand_texts)}
    enc(["warm-up"])
    E_te, case_ms = {}, {}
    for ci in range(len(test)):                      # online part: the five states of a case in one call
        group = [it for it in te if it[0] == ci]
        t0 = time.perf_counter()
        V = enc([it[2] for it in group])
        case_ms[ci] = 1000 * (time.perf_counter() - t0)
        for it, v in zip(group, V):
            E_te[(ci, it[1])] = v

    per_case = {ci: sum(1 for it in te if it[0] == ci) for ci in range(len(test))}

    def evaluate(h, name, extra=None):
        ZA = h.project(C, "action")
        preds, ms = [{} for _ in test], []
        for it in te:
            t0 = time.perf_counter()
            zs = h.project(E_te[(it[0], it[1])][None], "state")[0]
            s = ZA[[cid[t] for t in it[4]]] @ zs
            preds[it[0]][it[1]] = it[3][int(np.argmax(s))]
            ms.append(1000 * (time.perf_counter() - t0) + case_ms[it[0]] / per_case[it[0]])
        report(name, preds, ms, extra)

    if "clm8b" in todo:
        evaluate(clm.Heads(), "clm8b")
    if "clm8b-ft" in todo:
        ft_path = os.path.join(DATA, "clm", "heads_ft_typed.npz")
        info_key = "typed/train/clm8b-ft"
        if not os.path.exists(ft_path) or info_key not in r.res:
            S = cached_vectors(enc, [it[2] for it in tr], tag)
            kmax = max(len(it[4]) for it in tr)
            cand = -np.ones((len(tr), kmax), dtype=np.int64)
            for row, it in enumerate(tr):
                cand[row, :len(it[4])] = [cid[t] for t in it[4]]
            print(f"    training CLM-8B heads on {len(tr)} typed decisions", flush=True)
            info = clm.finetune(dict(S=S, C=C, cand_idx=cand, gold=np.array([it[5] for it in tr]),
                                     case_ids=np.array([it[0] for it in tr])), ft_path, grouped=True)
            r.res[info_key] = info
            save_json(r.out_name, r.res)
        evaluate(clm.Heads(ft_path), "clm8b-ft", {"train": r.res[info_key]})
    enc.close()
    del enc
    gc.collect()


# ---------------------------------------------------------------- aim: trial and error with refutations
AIM_PROMPT = dict(task="You are aiming at a hidden target on a grid; after each miss you are told in which direction "
                       "the target lies. Use every hint.", question="Which cell should be shot next?", noun="cell",
                  state_label="Game")


def aim_render(G, history):
    head = (f"Target practice on a {G}x{G} grid. Columns x go from 0 (left) to {G - 1} (right), rows y from 0 (top) "
            f"to {G - 1} (bottom). One cell hides the target.\nShots so far:\n")
    lines = [f"{i + 1}. x={x}, y={y} -> {fb}" for i, (x, y, fb) in enumerate(history)] or ["none yet"]
    return head + "\n".join(lines)


def aim_feedback(x, y, tx, ty):
    hx = "LEFT (smaller x)" if tx < x else "RIGHT (larger x)" if tx > x else "in this same column (same x)"
    hy = "LOWER (larger y)" if ty > y else "HIGHER (smaller y)" if ty < y else "in this same row (same y)"
    return f"miss: the target is {hx} and {hy}"


def aim_task(r: Runner):
    a = r.a
    G, games, max_shots = 15, 20, 25
    cells = [f"x={x}, y={y}" for y in range(G) for x in range(G)]
    rng = random.Random(1234)
    targets = [(rng.randrange(G), rng.randrange(G)) for _ in range(games)]
    todo = [m for m in METHODS["aim"] if r.want("aim", len(cells), m)]

    def play(name, player):
        won, shots_won, ms = 0, [], []
        for tx, ty in targets:
            history, tried = [], []
            for shot in range(1, max_shots + 1):
                t0 = time.perf_counter()
                choice = player(aim_render(G, history), tried)
                ms.append(1000 * (time.perf_counter() - t0))
                x, y = (int(v.split("=")[1]) for v in choice.split(", "))
                if (x, y) == (tx, ty):
                    won += 1
                    shots_won.append(shot)
                    break
                tried.append(choice)
                history.append((x, y, aim_feedback(x, y, tx, ty)))
        r.log("aim", len(cells), name, {"acc": won / games, "ms": statistics.median(ms), "games": games,
                                        "mean_shots_when_won": float(np.mean(shots_won)) if shots_won else None})

    if "random" in todo:
        rr = random.Random(0)
        play("random", lambda state, tried: rr.choice([c for c in cells if c not in tried]))
    if "kvindex" in todo or "openjev" in todo:
        kv = r.kv(a.n_ctx)
        be = kv.client
        if "kvindex" in todo:
            idx = kv.index(cells, **AIM_PROMPT)
            play("kvindex", lambda state, tried: idx.choose(state, exclude=tried).choice)
        if "openjev" in todo:
            def openjev_player(state, tried):
                remaining = [c for c in cells if c not in tried]          # OpenJev receives the remaining cells
                return openjev_pick(be, Index(kv, remaining, warm=False, **AIM_PROMPT), state)
            play("openjev", openjev_player)
        be.close()
        del kv, be
        gc.collect()
    if "clm8b" in todo:
        enc = LlamaCppEncoder(a.llm, pooling="last")
        heads = clm.Heads()
        za = heads.project(cached_vectors(enc, cells, "clm_" + a.llm.replace(":", "_")), "action")
        question = f'{AIM_PROMPT["task"]} {AIM_PROMPT["question"]}'
        pos = {c: i for i, c in enumerate(cells)}

        def clm_player(state, tried):
            zs = heads.project(enc([clm.state_text(f'{AIM_PROMPT["state_label"]}: {state}', question)]), "state")[0]
            s = za @ zs
            s[[pos[t] for t in tried]] = -np.inf
            return cells[int(np.argmax(s))]
        play("clm8b", clm_player)
        enc.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["wordnet", "banking77", "typed", "aim"])
    ap.add_argument("--llm", default="qwen3:8b")
    ap.add_argument("--embed", default="qwen3-embedding:4b")
    ap.add_argument("--kv-type", default="q8_0")
    ap.add_argument("--sizes", default="200,1000,3000,10000,30000")
    ap.add_argument("--queries", type=int, default=0, help="default: 150 (wordnet), 300 (banking77)")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--limit", type=int, default=0, help="typed: first N test cases only (smoke test)")
    ap.add_argument("--redo", action="store_true", help="recompute entries already in the results file")
    ap.add_argument("--n-ctx", type=int, default=16384, help="context of the typed and aim tasks")
    a = ap.parse_args()
    a.methods = METHODS[a.task] if a.methods == "all" else a.methods.split(",")
    a.queries = a.queries or {"wordnet": 150, "banking77": 300}.get(a.task, 0)
    r = Runner(a)
    print(f"task={a.task} llm={a.llm} kv={a.kv_type} methods={a.methods}", flush=True)
    if a.task == "wordnet":
        for n in (int(x) for x in a.sizes.split(",")):
            list_task(r, "wordnet", n)
    elif a.task == "banking77":
        list_task(r, "banking77", 0)
    elif a.task == "typed":
        typed_task(r)
    else:
        aim_task(r)


if __name__ == "__main__":
    main()
