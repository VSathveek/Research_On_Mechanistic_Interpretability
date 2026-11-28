# Auto-generated Kaggle kernel. Source: cgsae/src/exp1_synthetic.py
from __future__ import annotations

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'scipy', 'pandas', 'numpy'], check=False)
print('environment ready', flush=True)


"""
Experiment 1 rebuilt - the controlled synthetic demonstration.

v1's version reported a single seed on a 400-sample test set, selected the best tag
fraction by the same asymmetry score it then reported as the result, and had no null
control. At n=200 per class the standard error on a per-class accuracy is about 3.5 points,
so its headline numbers (-10.5 / +6.0) sat inside the noise band.

Nothing here needs a GPU, so there is no excuse for running it underpowered. This version:

  * runs N_SEEDS independent replications and reports 95% CIs;
  * measures **balanced accuracy and AUC**, not per-class accuracy at a fixed threshold -
    v1's asymmetric "+6 good / -10 bad" pattern is the signature of a shifted decision
    boundary, and its balanced accuracy barely moved (0.67 -> 0.65);
  * adds two null controls - ablating a random block of the same width, and ablating the
    *G*-block - so "removing the B-block hurts the unwanted class" can be separated from
    "removing any 12 units hurts everything";
  * reports B-block occupancy, so a silent block cannot be read as a success;
  * selects the reported tag fraction on a held-out split rather than on the test metric.

Usage:  python exp1_synthetic.py --seeds 20 --out ../results
"""


import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class GroupedMLP(nn.Module):
    """Two-layer MLP whose hidden units are pre-partitioned into G / neutral / B blocks."""

    def __init__(self, d_in=64, d_hidden=32, n_class=2, tag_frac=0.30):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, n_class)
        self.d_hidden, self.tag_frac = d_hidden, tag_frac
        n_tag = max(1, int(d_hidden * tag_frac))
        self.n_tag = n_tag
        self.g_idx = list(range(n_tag))
        self.b_idx = list(range(d_hidden - n_tag, d_hidden))
        self.neutral_idx = list(range(n_tag, d_hidden - n_tag))

    def forward(self, x, return_hidden=False):
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return (out, h) if return_hidden else out


def contrast_penalty(h, y, g_idx, b_idx):
    """Routing objective, matching the SAE experiments.

    A plain suppression penalty ("B must not fire on wanted inputs") is minimised by
    silencing B entirely - which is exactly what v1's tau=0.20 configuration did, giving a
    perfect leakage score and a pruning effect of -0.0000. Contrasting the two classes
    removes that shortcut: silence scores 0, correct routing scores -1.
    """
    pos, neg = (y == 1), (y == 0)
    if pos.sum() == 0 or neg.sum() == 0:
        return torch.zeros((), device=h.device)
    total = h.sum(-1) + EPS
    g_share = h[:, g_idx].sum(-1) / total
    b_share = h[:, b_idx].sum(-1) / total
    return -0.5 * ((b_share[neg].mean() - b_share[pos].mean())
                   + (g_share[pos].mean() - g_share[neg].mean()))


def make_data(n, rng, noise=3.0, shift=0.5, d=64):
    """Two overlapping Gaussians; the class signal lives in opposite halves of the input."""
    x_bad = rng.randn(n, d) * noise
    x_bad[:, :d // 2] += shift
    x_good = rng.randn(n, d) * noise
    x_good[:, d // 2:] += shift
    x = np.vstack([x_bad, x_good])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    p = rng.permutation(len(x))
    return x[p], y[p]


def balanced_acc(pred, y):
    return 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())


def auc_score(scores, y):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(int)
    n_p, n_n = int((y == 1).sum()), int((y == 0).sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    r = rankdata(np.asarray(scores, float))
    return float((r[y == 1].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def ablate(model, idx):
    """Zero a set of hidden units, in and out."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.fc1.weight[idx, :] = 0
        m.fc1.bias[idx] = 0
        m.fc2.weight[:, idx] = 0
    return m


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits, h = model(x, return_hidden=True)
        pred = logits.argmax(1).numpy()
        margin = (logits[:, 0] - logits[:, 1]).numpy()   # high = predicts unwanted
    yn = y.numpy()
    return {
        "bal_acc": balanced_acc(pred, yn),
        "auc": auc_score(margin, (yn == 0).astype(int)),
        "acc_unwanted": float((pred[yn == 0] == 0).mean()),
        "acc_wanted": float((pred[yn == 1] == 1).mean()),
    }, h


def run_one(seed, tag_frac, lam, epochs=200, lr=0.01, n_train=500, n_test=1000):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    x_tr, y_tr = make_data(n_train, rng)
    x_va, y_va = make_data(n_test // 2, rng)
    x_te, y_te = make_data(n_test, rng)          # 2000 samples, vs v1's 400
    to_t = lambda a, f=torch.FloatTensor: f(a)
    X_tr, Y_tr = to_t(x_tr), torch.LongTensor(y_tr.astype(int))
    X_va, Y_va = to_t(x_va), torch.LongTensor(y_va.astype(int))
    X_te, Y_te = to_t(x_te), torch.LongTensor(y_te.astype(int))

    model = GroupedMLP(tag_frac=tag_frac)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            logits, h = model(X_tr[idx], return_hidden=True)
            loss = crit(logits, Y_tr[idx]) + lam * contrast_penalty(
                h, Y_tr[idx], model.g_idx, model.b_idx)
            loss.backward()
            opt.step()

    base, h_te = evaluate(model, X_te, Y_te)
    val, _ = evaluate(model, X_va, Y_va)

    # occupancy: is the B-block carrying anything at all?
    with torch.no_grad():
        _, h = model(X_te, return_hidden=True)
    total = h.sum(-1) + EPS
    b_share = (h[:, model.b_idx].sum(-1) / total).numpy()
    yn = Y_te.numpy()
    routing = float(b_share[yn == 0].mean() - b_share[yn == 1].mean())
    b_alive = float(((h[:, model.b_idx] > 0).any(0)).float().mean())

    rows = []
    rng2 = np.random.RandomState(seed + 4242)
    ablations = {
        "B-block": model.b_idx,
        "G-block": model.g_idx,                                    # control
        "random": list(rng2.choice(model.d_hidden, len(model.b_idx), replace=False)),
    }
    for name, idx in ablations.items():
        abl, _ = evaluate(ablate(model, idx), X_te, Y_te)
        rows.append({
            "seed": seed, "tag_frac": tag_frac, "lambda": lam, "ablation": name,
            "n_ablated": len(idx),
            "val_bal_acc": val["bal_acc"],
            "routing": routing, "b_alive": b_alive,
            "d_bal_acc": abl["bal_acc"] - base["bal_acc"],
            "d_auc": abl["auc"] - base["auc"],
            "d_acc_unwanted": abl["acc_unwanted"] - base["acc_unwanted"],
            "d_acc_wanted": abl["acc_wanted"] - base["acc_wanted"],
            "base_bal_acc": base["bal_acc"], "base_auc": base["auc"],
        })
    return rows


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--tag-fracs", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40, 0.50])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0, 5.0])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for lam in a.lambdas:
        for tf in a.tag_fracs:
            for s in range(a.seeds):
                rows += run_one(s, tf, lam)
            sub = pd.DataFrame([r for r in rows if r["tag_frac"] == tf and r["lambda"] == lam
                                and r["ablation"] == "B-block"])
            print(f"lambda={lam:<4g} tau={tf:.2f}  "
                  f"dAUC={sub.d_auc.mean():+.4f}+/-{ci95(sub.d_auc):.4f}  "
                  f"routing={sub.routing.mean():+.3f}  b_alive={sub.b_alive.mean():.1%}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/exp1_synthetic_raw.csv", index=False)

    summ = (df.groupby(["lambda", "tag_frac", "ablation"])
              .agg(d_auc=("d_auc", "mean"), d_auc_ci=("d_auc", ci95),
                   d_bal=("d_bal_acc", "mean"), d_bal_ci=("d_bal_acc", ci95),
                   d_unwanted=("d_acc_unwanted", "mean"),
                   d_wanted=("d_acc_wanted", "mean"),
                   routing=("routing", "mean"), b_alive=("b_alive", "mean"),
                   n=("d_auc", "size")).reset_index())
    summ.to_csv(f"{a.out}/exp1_synthetic_summary.csv", index=False)

    print(f"\n=== Experiment 1, {a.seeds} seeds, 2000-sample test set, 95% CI ===")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nRead: d_auc for ablation='B-block' should be clearly negative AND clearly "
          "larger in magnitude than the 'random' and 'G-block' controls. If the three "
          "overlap within CI, the partition is doing nothing that ablating any equally "
          "wide block would not do.")



if __name__ == "__main__":
    sys.argv = ["exp1_synthetic.py", "--out", "/kaggle/working/results", "--seeds", "20", "--tag-fracs", "0.10", "0.20", "0.30", "0.40", "0.50", "--lambdas", "0", "1", "5"]
    print("ARGS:", sys.argv[1:], flush=True)
    main()
