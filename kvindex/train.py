"""
Optional training of the embedding catalogue.

A `Head` is a low-rank correction of the embedding space, one for states and one for options:
    project(X) = normalise(X + (X @ U) @ V.T)
With V = 0 it is the identity, so training starts from the untrained catalogue. `fit_head` learns U and V from
examples (state vector, index of the right option) with a softmax over the options, keeps the weights of the best
validation epoch and reports the validation accuracy before and after. Only the catalogue is trained; the LLM is not.
"""
from __future__ import annotations

import json
import time

import numpy as np


def _normalise(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


class Head:
    """Trained projection of state and option vectors (see the module docstring)."""

    def __init__(self, us, vs, uo, vo, report=None):
        self.us = np.asarray(us, dtype=np.float32)
        self.vs = np.asarray(vs, dtype=np.float32)
        self.uo = np.asarray(uo, dtype=np.float32)
        self.vo = np.asarray(vo, dtype=np.float32)
        self.report = dict(report or {})

    def state(self, m) -> np.ndarray:
        """Project state vectors (n, d) -> (n, d), L2-normalised."""
        x = np.asarray(m, dtype=np.float32)
        return _normalise(x + (x @ self.us) @ self.vs.T)

    def options(self, m) -> np.ndarray:
        """Project option vectors (m, d) -> (m, d), L2-normalised."""
        x = np.asarray(m, dtype=np.float32)
        return _normalise(x + (x @ self.uo) @ self.vo.T)

    def save(self, path: str) -> None:
        """Write the head to `path` (a .npz file)."""
        np.savez(path, us=self.us, vs=self.vs, uo=self.uo, vo=self.vo, report=json.dumps(self.report))

    @classmethod
    def load(cls, path: str) -> "Head":
        """Read a head written by `save`."""
        z = np.load(path)
        return cls(z["us"], z["vs"], z["uo"], z["vo"], json.loads(str(z["report"])))


def loss_and_grads(p: dict, xs: np.ndarray, xo: np.ndarray, pos: np.ndarray, scale: float) -> tuple[float, dict]:
    """Mean cross-entropy of a batch and its gradients with respect to us, vs, uo, vo (float64 in, float64 out)."""
    hs = xs @ p["us"]
    ys = xs + hs @ p["vs"].T
    ns = np.linalg.norm(ys, axis=1, keepdims=True) + 1e-12
    zs = ys / ns

    ho = xo @ p["uo"]
    yo = xo + ho @ p["vo"].T
    no = np.linalg.norm(yo, axis=1, keepdims=True) + 1e-12
    zo = yo / no

    logits = scale * (zs @ zo.T)
    logits = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    prob = prob / prob.sum(axis=1, keepdims=True)

    g = prob.copy()
    g[np.arange(len(xs)), pos] -= 1.0
    g = g / len(xs)

    dzs = scale * (g @ zo)
    dzo = scale * (g.T @ zs)

    dys = (dzs - zs * np.sum(dzs * zs, axis=1, keepdims=True)) / ns
    dyo = (dzo - zo * np.sum(dzo * zo, axis=1, keepdims=True)) / no

    grads = {
        "vs": dys.T @ hs,
        "us": xs.T @ (dys @ p["vs"]),
        "vo": dyo.T @ ho,
        "uo": xo.T @ (dyo @ p["vo"])
    }

    loss = float(-np.mean(np.log(prob[np.arange(len(xs)), pos] + 1e-30)))

    return (loss, grads)


def fit_head(states, options, ids, rank: int = 64, epochs: int = 40, batch: int = 256, lr: float = 1e-3,
             scale: float = 30.0, val_frac: float = 0.1, negatives: int = 1024, seed: int = 0) -> Head:
    """Train a Head. `states`: (n, d) vectors of the example states; `options`: (m, d) vectors of the options;
    `ids`: for every example, the position of its right option. Returns the Head of the best validation epoch;
    `head.report` has val_start, val_best, best_epoch, train_s, train_examples, val_examples."""
    S = np.asarray(states, dtype=np.float32)
    O = np.asarray(options, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int64)
    n, d = S.shape
    m = len(O)
    rng = np.random.default_rng(seed)

    if n < 2:
        raise ValueError("training needs at least 2 examples")

    n_val = max(1, int(val_frac * n))
    perm = rng.permutation(n)
    val = perm[:n_val]
    tr = perm[n_val:]

    p = {
        "us": rng.normal(0, 0.01, (d, rank)).astype(np.float32),
        "vs": np.zeros((d, rank), dtype=np.float32),
        "uo": rng.normal(0, 0.01, (d, rank)).astype(np.float32),
        "vo": np.zeros((d, rank), dtype=np.float32)
    }

    m1 = {k: np.zeros_like(v) for k, v in p.items()}
    m2 = {k: np.zeros_like(v) for k, v in p.items()}
    t = 0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    def accuracy():
        zs = _normalise(S[val] + (S[val] @ p["us"]) @ p["vs"].T)
        zo = _normalise(O + (O @ p["uo"]) @ p["vo"].T)
        return float(np.mean(np.argmax(zs @ zo.T, axis=1) == ids[val]))

    best_acc = accuracy()
    val_start = best_acc
    best_epoch = 0
    best = {k: v.copy() for k, v in p.items()}
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        order = tr[rng.permutation(len(tr))]
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            if m <= 4096:
                cand = np.arange(m)
            else:
                cand = np.unique(np.concatenate([ids[b], rng.integers(0, m, negatives)]))
            pos = np.searchsorted(cand, ids[b])
            loss, grads = loss_and_grads(p, S[b], O[cand], pos, scale)
            t += 1
            for k in grads:
                m1[k] = beta1 * m1[k] + (1 - beta1) * grads[k]
                m2[k] = beta2 * m2[k] + (1 - beta2) * grads[k] ** 2
                p[k] = (p[k] - lr * (m1[k] / (1 - beta1 ** t)) /
                        (np.sqrt(m2[k] / (1 - beta2 ** t)) + eps)).astype(np.float32)
        acc = accuracy()
        if acc > best_acc:
            best_acc = acc
            best_epoch = ep
            best = {k: v.copy() for k, v in p.items()}

    report = {
        "val_start": float(val_start),
        "val_best": float(best_acc),
        "best_epoch": int(best_epoch),
        "train_s": float(time.perf_counter() - t0),
        "train_examples": int(len(tr)),
        "val_examples": int(len(val)),
        "n_options": int(m),
        "trained_options": sorted(int(i) for i in set(ids.tolist()))
    }
    return Head(best["us"], best["vs"], best["uo"], best["vo"], report)


import numpy as np
import time

def grouped_loss_and_grads(p, xs, xo, cand, tgt, scale):
    """Cross-entropy of soft targets over per-example candidates, and its gradients (float64 in, float64 out).

    p: dict with us, vs, uo, vo, each (d, r). xs: (B, d) state vectors of the batch. xo: (U, d) vectors of the
    options used by the batch. cand: (B, K) int, for each example the rows of `xo` of its candidates, -1 where the
    example has fewer than K candidates. tgt: (B, K) float, the target distribution over the candidates (0 at -1)."""
    hs = xs @ p["us"]
    ys = xs + hs @ p["vs"].T
    ns = np.linalg.norm(ys, axis=1, keepdims=True) + 1e-12
    zs = ys / ns
    ho = xo @ p["uo"]
    yo = xo + ho @ p["vo"].T
    no = np.linalg.norm(yo, axis=1, keepdims=True) + 1e-12
    zo = yo / no
    valid = cand >= 0
    safe = np.where(valid, cand, 0)
    logits = scale * np.einsum("bd,bkd->bk", zs, zo[safe])
    logits = np.where(valid, logits, -np.inf)
    logits = logits - logits.max(axis=1, keepdims=True)
    prob = np.where(valid, np.exp(logits), 0.0)
    prob = prob / prob.sum(axis=1, keepdims=True)
    B = len(xs)
    loss = float(-np.sum(tgt * np.log(prob + 1e-30)) / B)
    g = (prob - tgt) / B
    g = np.where(valid, g, 0.0)
    dzs = scale * np.einsum("bk,bkd->bd", g, zo[safe])
    dzo = np.zeros_like(zo)
    contrib = (scale * g[:, :, None] * zs[:, None, :]).reshape(-1, zs.shape[1])
    np.add.at(dzo, safe.ravel(), contrib)
    dys = (dzs - zs * np.sum(dzs * zs, axis=1, keepdims=True)) / ns
    dyo = (dzo - zo * np.sum(dzo * zo, axis=1, keepdims=True)) / no
    grads = {"vs": dys.T @ hs, "us": xs.T @ (dys @ p["vs"]), "vo": dyo.T @ ho, "uo": xo.T @ (dyo @ p["vo"])}
    return loss, grads


def fit_head_grouped(states, options, candidates, targets, rank: int = 64, epochs: int = 40, batch: int = 256,
                     lr: float = 1e-3, scale: float = 30.0, val_frac: float = 0.1, seed: int = 0) -> Head:
    """Train a Head on examples with their own candidate options and soft targets. `states`: (n, d);
    `options`: (m, d), one row per distinct option text; `candidates`: list of n integer lists (rows of
    `options`); `targets`: list of n float lists, the target distribution over each example's candidates (a
    one-hot list for a hard label). Keeps the weights of the best validation epoch; the validation accuracy is
    the share of examples whose most probable candidate is the target's most probable one."""
    S = np.asarray(states, dtype=np.float32)
    O = np.asarray(options, dtype=np.float32)
    n, d = S.shape
    if n < 2:
        raise ValueError("training needs at least 2 examples")
    if len(candidates) != n or len(targets) != n:
        raise ValueError("one candidate list and one target per example")
    K = max(len(c) for c in candidates)
    cand = -np.ones((n, K), dtype=np.int64)
    tgt = np.zeros((n, K), dtype=np.float64)
    for i, (c, t) in enumerate(zip(candidates, targets)):
        if len(c) != len(t):
            raise ValueError(f"example {i}: {len(c)} candidates but {len(t)} target values")
        cand[i, :len(c)] = c
        tt = np.asarray(t, dtype=np.float64)
        tgt[i, :len(c)] = tt / tt.sum()
    gold = np.argmax(np.where(cand >= 0, tgt, -1.0), axis=1)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(val_frac * n))
    perm = rng.permutation(n)
    val = perm[:n_val]
    tr = perm[n_val:]
    p = {"us": rng.normal(0, 0.01, (d, rank)).astype(np.float32),
         "vs": np.zeros((d, rank), dtype=np.float32),
         "uo": rng.normal(0, 0.01, (d, rank)).astype(np.float32),
         "vo": np.zeros((d, rank), dtype=np.float32)}
    m1 = {k: np.zeros_like(v) for k, v in p.items()}
    m2 = {k: np.zeros_like(v) for k, v in p.items()}
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    def accuracy():
        zs = _normalise(S[val] + (S[val] @ p["us"]) @ p["vs"].T)
        zo = _normalise(O + (O @ p["uo"]) @ p["vo"].T)
        c = cand[val]
        valid = c >= 0
        sc = np.where(valid, np.einsum("bd,bkd->bk", zs, zo[np.where(valid, c, 0)]), -np.inf)
        return float(np.mean(np.argmax(sc, axis=1) == gold[val]))
    
    best_acc = accuracy()
    val_start = best_acc
    best_epoch = 0
    best = {k: v.copy() for k, v in p.items()}
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1): 
        order = tr[rng.permutation(len(tr))]
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            cb = cand[b]
            used = np.unique(cb[cb >= 0])
            local = np.searchsorted(used, np.where(cb >= 0, cb, used[0]))
            local = np.where(cb >= 0, local, -1)
            loss, grads = grouped_loss_and_grads(p, S[b], O[used], local, tgt[b], scale)
            t += 1
            for k in grads:
                m1[k] = beta1 * m1[k] + (1 - beta1) * grads[k]
                m2[k] = beta2 * m2[k] + (1 - beta2) * grads[k] ** 2
                p[k] = (p[k] - lr * (m1[k] / (1 - beta1 ** t)) /
                        (np.sqrt(m2[k] / (1 - beta2 ** t)) + eps)).astype(np.float32)
        acc = accuracy()
        if acc > best_acc:
            best_acc = acc
            best_epoch = ep
            best = {k: v.copy() for k, v in p.items()}
    report = {"val_start": float(val_start), "val_best": float(best_acc), "best_epoch": int(best_epoch),
              "train_s": float(time.perf_counter() - t0), "train_examples": int(len(tr)),
              "val_examples": int(len(val)),
              "scale": float(scale)}
    return Head(best["us"], best["vs"], best["uo"], best["vo"], report)
