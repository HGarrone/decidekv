"""
CLM-8B (Contrastive Language Models, release Contrastive-LM/CLM-v0.1-8B) as a reference method.

* Encoder: Qwen3-8B, hidden state of the last token after the final norm, L2-normalised. Computed with llama.cpp from
  the same GGUF file used by the LLM methods (the reference stack is vLLM in bf16; the GGUF is quantised).
* Heads: the released state and action projection heads, evaluated in numpy.
  score(state, option) = cos(state_head(e_state), action_head(e_option)).
* Fine-tuning ("clm8b-ft"): the released heads are trained further on a task's training split; the epoch with the
  best validation accuracy is kept (epoch 0 = the released heads). PyTorch runs in a child process: PyTorch and
  llama.cpp ship different OpenMP runtimes that cannot share one process on Windows.

    python benchmarks/clm.py export          # download the released checkpoint, export the heads to numpy (needs torch)
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATA  # noqa: E402

HF_URL = "https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/resolve/main/CLM_v0.1-8B.pt"
CKPT = os.path.join(DATA, "clm", "CLM_v0.1-8B.pt")
HEADS = os.path.join(DATA, "clm", "heads_np.npz")


def state_text(context: str, question: str) -> str:
    """CLM's input layout for a state: the context, a blank line, the question."""
    context, question = context.strip(), (question or "").strip()
    return f"{context}\n\n{question}" if context and question else (context or question)


# ---------------------------------------------------------------- heads in numpy
try:
    from scipy.special import erf as _erf
except ImportError:                                    # pragma: no cover
    import math
    _erf = np.vectorize(math.erf)


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))   # exact GELU, as torch.nn.GELU()


class Heads:
    """State/action heads stored as numpy arrays (released or fine-tuned)."""

    def __init__(self, path: str = HEADS):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found: run `python benchmarks/clm.py export` first")
        z = np.load(path)
        self.w = {k: z[k] for k in z.files}
        self.depth = int(self.w["depth"])
        self.scale = float(self.w["scale"])

    def project(self, e, side: str) -> np.ndarray:
        w, p = self.w, side + "."
        x = np.asarray(e, dtype=np.float32)
        h = _gelu(x @ w[p + "inp.weight"].T + w[p + "inp.bias"])
        for i in range(self.depth - 2):
            z = h @ w[p + f"hidden.{i}.weight"].T + w[p + f"hidden.{i}.bias"]
            if p + f"norms.{i}.weight" in w:                                   # LayerNorm (eps 1e-5)
                z = (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + 1e-5)
                z = z * w[p + f"norms.{i}.weight"] + w[p + f"norms.{i}.bias"]
            h = _gelu(z)
        out = h @ w[p + "out.weight"].T + w[p + "out.bias"]
        return (out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-12)).astype(np.float32)


# ---------------------------------------------------------------- fine-tuning (child process with torch)
def finetune(arrays: dict, out_path: str, grouped: bool = False, start: str = HEADS) -> dict:
    """Train the heads further. `arrays`: S, A, ids, split_by_action (closed or open option sets), or, with
    grouped=True, S, C, cand_idx, gold, case_ids (every example has its own small option set). Writes the
    fine-tuned heads to `out_path` (numpy) and returns the training report."""
    with tempfile.TemporaryDirectory() as d:
        req = os.path.join(d, "req.npz")
        info = os.path.join(d, "info.json")
        np.savez(req, **arrays)
        cmd = [sys.executable, os.path.abspath(__file__), "train-grouped" if grouped else "train", req, start,
               out_path, info]
        subprocess.run(cmd, check=True)
        return json.load(open(info))


def _torch_heads(path):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    z = np.load(path)
    depth = int(z["depth"])

    class Head(nn.Module):
        def __init__(self, side):
            super().__init__()
            g = lambda k: torch.from_numpy(z[f"{side}.{k}"].copy())        # noqa: E731
            self.inp = nn.Linear(*g("inp.weight").shape[::-1])
            self.hidden = nn.ModuleList(nn.Linear(*g(f"hidden.{i}.weight").shape[::-1]) for i in range(depth - 2))
            ln = f"{side}.norms.0.weight" in z.files
            width = g("inp.weight").shape[0]
            self.norms = nn.ModuleList((nn.LayerNorm(width) if ln else nn.Identity()) for _ in range(depth - 2))
            self.out = nn.Linear(*g("out.weight").shape[::-1])
            self.act = nn.GELU()
            state = {k[len(side) + 1:]: torch.from_numpy(z[k].copy()) for k in z.files if k.startswith(side + ".")}
            self.load_state_dict(state)

        def forward(self, x):
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                x = self.act(nrm(lin(x)))
            return self.out(x)

    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.s, self.a = Head("state"), Head("action")
            self.logit_scale = nn.Parameter(torch.tensor(float(np.log(float(z["scale"]))), dtype=torch.float32))

        def fs(self, x):
            return F.normalize(self.s(x), dim=-1)

        def fa(self, x):
            return F.normalize(self.a(x), dim=-1)

        def save(self, out):
            arrays = {"scale": np.array(float(self.logit_scale.exp().clamp(max=100)), dtype=np.float32),
                      "depth": np.array(depth)}
            for side, head in (("state", self.s), ("action", self.a)):
                for k, v in head.state_dict().items():
                    arrays[f"{side}.{k}"] = v.detach().numpy().astype(np.float32)
            np.savez(out, **arrays)

    return Pair()


def _train(req, start, out, info, epochs=40, bs=256, lr=1e-3, n_hard=8, val_frac=0.1, seed=0):
    """State -> option cross-entropy over all options (closed sets) or over in-batch options plus hard negatives
    mined with the released heads (open sets). split_by_action holds out whole options (unseen options at test
    time, e.g. new words); otherwise it holds out states of known options (e.g. intents)."""
    import copy
    import torch
    import torch.nn.functional as F
    z = np.load(req)
    S, A, ids, split_by_action = z["S"], z["A"], z["ids"], bool(z["split_by_action"])
    model = _torch_heads(start)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n, m = len(S), len(A)
    if split_by_action:
        val_actions = set(rng.choice(m, max(1, int(val_frac * m)), replace=False).tolist())
        val = np.array([i for i in range(n) if ids[i] in val_actions])
    else:
        val = rng.choice(n, max(1, int(val_frac * n)), replace=False)
    tr = np.setdiff1d(np.arange(n), val)
    St, At = torch.from_numpy(S), torch.from_numpy(A)
    if split_by_action:
        pool = np.array(sorted({int(ids[i]) for i in val}))
        remap = {a: j for j, a in enumerate(pool)}
        v_ids, v_A = torch.tensor([remap[int(ids[i])] for i in val]), At[pool]
    else:
        v_ids, v_A = torch.from_numpy(ids[val]).long(), At
    v_S = St[val]

    def accuracy():
        with torch.no_grad():
            return float(((model.fs(v_S) @ model.fa(v_A).T).argmax(1) == v_ids).float().mean())

    full = m <= 4096
    if not full:
        train_actions = np.array(sorted({int(ids[i]) for i in tr}))
        with torch.no_grad():
            zs_all, za_tr = model.fs(St), model.fa(At[train_actions])
        hard = np.zeros((n, n_hard), dtype=np.int64)
        for i in range(0, len(tr), 2048):
            rows = tr[i:i + 2048]
            top = (zs_all[rows] @ za_tr.T).topk(n_hard + 1, dim=1).indices.numpy()
            for r, row in zip(rows, train_actions[top]):
                hard[r] = [a for a in row if a != ids[r]][:n_hard]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best = (accuracy(), 0, copy.deepcopy(model.state_dict()))
    first = best[0]
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = tr[rng.permutation(len(tr))]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            cand = np.arange(m) if full else np.unique(np.concatenate([ids[b], hard[b].ravel()]))
            pos = {int(a): j for j, a in enumerate(cand)}
            target = torch.tensor([pos[int(ids[x])] for x in b])
            logits = model.logit_scale.exp().clamp(max=100) * model.fs(St[b]) @ model.fa(At[torch.from_numpy(cand)]).T
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        acc = accuracy()
        if acc > best[0]:
            best = (acc, ep, copy.deepcopy(model.state_dict()))
        print(f"      epoch {ep:>2}  loss {loss.item():.3f}  val acc {acc:.3f}", flush=True)
    model.load_state_dict(best[2])
    model.save(out)
    json.dump({"val_start": first, "val_best": best[0], "best_epoch": best[1], "train_s": time.perf_counter() - t0,
               "train_examples": int(len(tr)), "val_examples": int(len(val))}, open(info, "w"))


def _train_grouped(req, start, out, info, epochs=40, bs=128, lr=1e-3, val_frac=0.1, seed=0):
    """Every example has its own small option set (typed questions); validation on held-out cases."""
    import copy
    import torch
    import torch.nn.functional as F
    z = np.load(req)
    S, C, cand_idx, gold, case_ids = z["S"], z["C"], z["cand_idx"], z["gold"], z["case_ids"]
    model = _torch_heads(start)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    cases = np.array(sorted(set(case_ids.tolist())))
    val_cases = set(rng.choice(cases, max(1, int(val_frac * len(cases))), replace=False).tolist())
    is_val = np.array([c in val_cases for c in case_ids.tolist()])
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    St, Ct = torch.from_numpy(S), torch.from_numpy(C)
    idx = torch.from_numpy(np.where(cand_idx < 0, 0, cand_idx)).long()
    pad = torch.from_numpy(cand_idx < 0)
    gt = torch.from_numpy(gold).long()

    def logits(rows):
        zs, za = model.fs(St[rows]), model.fa(Ct)
        lg = model.logit_scale.exp().clamp(max=100) * torch.einsum("bp,bkp->bk", zs, za[idx[rows]])
        return lg.masked_fill(pad[rows], -1e9)

    def accuracy(rows):
        with torch.no_grad():
            return float((logits(rows).argmax(1) == gt[rows]).float().mean())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best = (accuracy(va), 0, copy.deepcopy(model.state_dict()))
    first = best[0]
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = tr[rng.permutation(len(tr))]
        for i in range(0, len(perm), bs):
            b = torch.from_numpy(perm[i:i + bs])
            loss = F.cross_entropy(logits(b), gt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        acc = accuracy(va)
        if acc > best[0]:
            best = (acc, ep, copy.deepcopy(model.state_dict()))
        print(f"      epoch {ep:>2}  loss {loss.item():.3f}  val acc {acc:.3f}", flush=True)
    model.load_state_dict(best[2])
    model.save(out)
    json.dump({"val_start": first, "val_best": best[0], "best_epoch": best[1], "train_s": time.perf_counter() - t0,
               "train_examples": int(len(tr)), "val_examples": int(len(va))}, open(info, "w"))


def export():
    """Download the released checkpoint and store its heads as numpy arrays."""
    import urllib.request
    import torch
    if not os.path.exists(CKPT):
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        print(f"downloading {HF_URL}")
        urllib.request.urlretrieve(HF_URL, CKPT)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    cfg = dict(ck["cfg"])
    if cfg.get("activation", "gelu") != "gelu" or cfg.get("residual", False):
        raise ValueError(f"unexpected head configuration {cfg}")
    arrays = {"scale": np.array(float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))),
              "depth": np.array(int(cfg["depth"]))}
    for side in ("state", "action"):
        for k, v in ck[f"{side}_head"].items():
            arrays[f"{side}.{k}"] = v.float().numpy()
    np.savez(HEADS, **arrays)
    print(f"saved {HEADS} (depth {cfg['depth']}, width {cfg['width']}, layernorm {cfg.get('layernorm', False)})")


def selftest(path: str = HEADS):
    """The numpy heads must reproduce the PyTorch heads (needs torch; no llama.cpp in this process)."""
    import torch
    x = np.random.default_rng(0).standard_normal((16, 4096)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    h, t = Heads(path), _torch_heads(path).eval()
    with torch.no_grad():
        ref = {"state": t.fs(torch.from_numpy(x)).numpy(), "action": t.fa(torch.from_numpy(x)).numpy()}
    ok = True
    for side in ("state", "action"):
        d = float(np.abs(h.project(x, side) - ref[side]).max())
        ok &= d < 1e-4
        print(f"[{'PASS' if d < 1e-4 else 'FAIL'}] numpy {side} head == torch (max |diff| = {d:.2e})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "export":
        export()
    elif cmd == "selftest":
        selftest(*sys.argv[2:3])
    elif cmd == "train":
        _train(*sys.argv[2:6])
    elif cmd == "train-grouped":
        _train_grouped(*sys.argv[2:6])
    else:
        print(__doc__)
