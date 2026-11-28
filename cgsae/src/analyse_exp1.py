"""
Figures and tables for the rebuilt Experiment 1.

The headline quantity is **specificity**:

    specificity = |dAUC when the B-block is ablated| / |dAUC when a random block of the
                   same width is ablated|

A value of 1 means the tagged block is no better than any equally wide slice of the
network - i.e. the partition did nothing. This ratio is the control v1 never ran, and it
is what separates "removing these units hurts the unwanted class" from "removing this many
units hurts everything".

Usage:  python analyse_exp1.py --results ../results --figures ../figures
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D_HIDDEN = 32


def buffer_ratio(tau: float, d_hidden: int = D_HIDDEN) -> float:
    n_tag = max(1, int(d_hidden * tau))
    return (d_hidden - 2 * n_tag) / (2 * n_tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    os.makedirs(a.figures, exist_ok=True)

    s = pd.read_csv(f"{a.results}/exp1_synthetic_summary.csv")
    piv = s.pivot_table(index=["lambda", "tag_frac"], columns="ablation",
                        values="d_auc").reset_index()
    piv["buffer_ratio"] = piv["tag_frac"].map(buffer_ratio)
    piv["specificity"] = piv["B-block"] / piv["random"]
    piv.to_csv(f"{a.results}/exp1_specificity.csv", index=False)

    # -- Fig A: specificity vs buffer ---------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colours = {0.0: "#999999", 1.0: "#C44E52", 5.0: "#4C72B0"}
    for lam, g in piv.groupby("lambda"):
        g = g.sort_values("tag_frac")
        ax.plot(g["tag_frac"], g["specificity"], marker="o", lw=2,
                color=colours.get(lam, None),
                label=("no penalty (lambda=0)" if lam == 0 else f"lambda = {lam:g}"))
    ax.axhline(1.0, color="black", ls="--", lw=1.2)
    ax.text(0.5, 1.0, " no specificity", va="bottom", ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("tag fraction tau   (0.5 = no neutral buffer)")
    ax.set_ylabel("specificity  =  |dAUC B-block| / |dAUC random block|")
    ax.set_title("The neutral buffer is what makes the tagged block specific\n"
                 "synthetic MLP, 20 seeds, 2000-sample test set", fontsize=10)
    sec = ax.secondary_xaxis("top", functions=(lambda t: t, lambda t: t))
    sec.set_xticks(sorted(piv["tag_frac"].unique()))
    sec.set_xticklabels([f"{buffer_ratio(t):.1f}" for t in sorted(piv["tag_frac"].unique())],
                        fontsize=8)
    sec.set_xlabel("buffer ratio  (neutral units per tagged unit)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{a.figures}/exp1_fig1_specificity.png", dpi=180)
    plt.close(fig)

    # -- Fig B: the double dissociation --------------------------------------
    best = s[(s["lambda"] == 1.0) & (np.isclose(s["tag_frac"], 0.20))]
    if not best.empty:
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        order = ["B-block", "G-block", "random"]
        w = 0.36
        x = np.arange(len(order))
        unw = [best[best.ablation == o]["d_unwanted"].iloc[0] for o in order]
        wan = [best[best.ablation == o]["d_wanted"].iloc[0] for o in order]
        ax.bar(x - w / 2, unw, w, label="unwanted-class accuracy", color="#C44E52")
        ax.bar(x + w / 2, wan, w, label="wanted-class accuracy", color="#4C72B0")
        for xi, (u, v) in enumerate(zip(unw, wan)):
            ax.text(xi - w / 2, u, f"{u:+.2f}", ha="center",
                    va="top" if u < 0 else "bottom", fontsize=8)
            ax.text(xi + w / 2, v, f"{v:+.2f}", ha="center",
                    va="top" if v < 0 else "bottom", fontsize=8)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["ablate B-block", "ablate G-block", "ablate random\n(same width)"])
        ax.set_ylabel("change in accuracy after ablation")
        ax.set_title("Double dissociation with a size-matched null\n"
                     "lambda=1, tau=0.20, 20 seeds", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(f"{a.figures}/exp1_fig2_dissociation.png", dpi=180)
        plt.close(fig)

    print("=== specificity (B-block vs size-matched random block) ===")
    show = piv[["lambda", "tag_frac", "buffer_ratio", "B-block", "random", "specificity"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nfigures -> {a.figures}/exp1_fig1_specificity.png, exp1_fig2_dissociation.png")


if __name__ == "__main__":
    main()
