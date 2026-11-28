"""
CG-SAE v2 - turn the run CSVs into the paper's figures and tables.

The headline comparison is *effect at a matched perplexity budget*: every method is swept
over intervention strength, and each arm is scored by how much unwanted behaviour it
removes at the point where it has spent the same amount of fluency. This is the number v1
could not report, because it measured a single operating point per method and so could not
tell "steers better" apart from "steers less".

Usage
-----
    python analyse.py --results results --figures figures --tag sentiment_gpt2_L8
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PPL_BUDGETS = (5.0, 10.0, 25.0)

ARM_STYLE = {
    "vanilla":       dict(color="#4C72B0", marker="o", label="Vanilla TopK SAE + post-hoc (SAE-SSV)"),
    "scar":          dict(color="#DD8452", marker="s", label="Scar (1 conditioned latent)"),
    "cgsae":         dict(color="#C44E52", marker="^", label="CG-SAE (group-restricted)"),
    "cgsae_free":    dict(color="#8172B3", marker="v", label="CG-SAE (post-hoc, ignores groups)"),
    "diff_in_means": dict(color="#55A868", marker="D", label="Difference-in-means / CAA"),
    "probe":         dict(color="#937860", marker="P", label="Logistic-probe direction"),
    "random_dir":    dict(color="#999999", marker="x", label="Random direction (null)"),
    "random":        dict(color="#CCCCCC", marker="+", label="Random latents (null)"),
}


def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return 0.0
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def arm_family(arm: str) -> str:
    """Collapse 'cgsae@0.10/group' -> a plotting family."""
    base = arm.split("/")[0]
    variant = arm.split("/")[1] if "/" in arm else ""
    if variant == "random":
        return "random"
    if base.startswith("cgsae"):
        return "cgsae" if variant == "group" else "cgsae_free"
    if base.startswith("vanilla"):
        return "vanilla"
    if base.startswith("scar"):
        return "scar"
    return base


def effect_at_budget(sub: pd.DataFrame, budget: float) -> float:
    """Interpolate an arm's effect at a given perplexity cost.

    Monotone interpolation over the arm's own strength sweep; NaN if the arm never reaches
    the budget (it cannot spend that much fluency) - which is itself informative.
    """
    s = sub.sort_values("ppl_pct")
    x, y = s["ppl_pct"].values, s["effect"].values
    if len(x) < 2 or budget < x.min():
        return float("nan")
    if budget >= x.max():
        return float(y[np.argmax(x)])
    return float(np.interp(budget, x, np.maximum.accumulate(y)))


# ---------------------------------------------------------------------------
def fig_tradeoff(steer: pd.DataFrame, out: str, tag: str) -> None:
    """Fig 1 - the headline: steering effect against perplexity cost."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    steer = steer.copy()
    steer["family"] = steer["arm"].map(arm_family)

    # For CG-SAE families keep the tag fraction with the best area under the curve.
    best_cg = {}
    for fam in ("cgsae", "cgsae_free"):
        sub = steer[steer["family"] == fam]
        if sub.empty:
            continue
        scores = {}
        for arm, g in sub.groupby("arm"):
            agg = g.groupby("alpha")[["effect", "ppl_pct"]].mean()
            scores[arm] = effect_at_budget(agg.reset_index(), 10.0)
        best_cg[fam] = max(scores, key=lambda k: (scores[k] if not np.isnan(scores[k]) else -9))

    for fam, style in ARM_STYLE.items():
        sub = steer[steer["family"] == fam]
        if fam in best_cg:
            sub = sub[sub["arm"] == best_cg[fam]]
        if sub.empty:
            continue
        agg = sub.groupby("alpha").agg(
            effect=("effect", "mean"), effect_ci=("effect", ci95),
            ppl=("ppl_pct", "mean")).reset_index().sort_values("ppl")
        lab = style["label"]
        if fam in best_cg:
            tf = best_cg[fam].split("@")[1].split("/")[0]
            lab += f" [tau={tf}]"
        ax.errorbar(agg["ppl"], agg["effect"], yerr=agg["effect_ci"],
                    color=style["color"], marker=style["marker"], label=lab,
                    lw=2, ms=6, capsize=3, alpha=0.9)

    for b in PPL_BUDGETS:
        ax.axvline(b, color="grey", ls=":", lw=1, alpha=0.6)
        ax.text(b, ax.get_ylim()[1], f" {b:.0f}%", color="grey", fontsize=8, va="top")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Perplexity cost on off-axis held-out text (%)")
    ax.set_ylabel("Reduction in unwanted behaviour (judge score)")
    ax.set_title(f"Steering effect vs fluency cost - {tag}\n"
                 "up and to the left is better; error bars are 95% CI over seeds",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/fig1_tradeoff_{tag}.png", dpi=180)
    plt.close(fig)


def fig_buffer_sweep(metrics: pd.DataFrame, out: str, tag: str) -> None:
    """Fig 2 - the novelty claim: what the neutral buffer buys and what it costs."""
    cg = metrics[metrics["kind"] == "cgsae"].copy()
    if cg.empty:
        return
    agg = cg.groupby("tag_frac").agg(
        leak=("norm_leakage", "mean"), leak_ci=("norm_leakage", ci95),
        fvu=("fvu", "mean"), fvu_ci=("fvu", ci95),
        buf=("buffer_ratio", "mean")).reset_index().sort_values("tag_frac")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.errorbar(agg["tag_frac"], agg["leak"], yerr=agg["leak_ci"],
                color="#C44E52", marker="o", lw=2, capsize=3, label="Normalised leakage")
    ax.set_xlabel("tag fraction tau  (0.5 = no neutral buffer)")
    ax.set_ylabel("Normalised leakage (lower = better separation)", color="#C44E52")
    ax.tick_params(axis="y", labelcolor="#C44E52")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.errorbar(agg["tag_frac"], agg["fvu"], yerr=agg["fvu_ci"],
                 color="#4C72B0", marker="s", lw=2, ls="--", capsize=3, label="FVU")
    ax2.set_ylabel("FVU (lower = better reconstruction)", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")

    van = metrics[metrics["kind"] == "vanilla"]["fvu"].mean()
    if not np.isnan(van):
        ax2.axhline(van, color="#4C72B0", ls=":", lw=1.5, alpha=0.7)
        ax2.text(agg["tag_frac"].max(), van, " vanilla SAE FVU",
                 color="#4C72B0", fontsize=8, va="bottom", ha="right")

    ax.set_title(f"Neutral buffer: separation vs reconstruction - {tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/fig2_buffer_sweep_{tag}.png", dpi=180)
    plt.close(fig)


def fig_metric_gameable(metrics: pd.DataFrame, out: str, tag: str) -> None:
    """Fig 3 - why v1's leakage number could not be trusted."""
    cg = metrics[metrics["kind"] == "cgsae"].copy()
    if cg.empty or "raw_leakage" not in cg:
        return
    agg = cg.groupby("tag_frac").agg(
        raw=("raw_leakage", "mean"), norm=("norm_leakage", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    w = 0.012
    ax.bar(agg["tag_frac"] - w / 2, agg["raw"], width=w, color="#DD8452",
           label="raw leakage (v1 metric)")
    ax.bar(agg["tag_frac"] + w / 2, agg["norm"], width=w, color="#4C72B0",
           label="normalised leakage (v2)")
    ax.set_xlabel("tag fraction tau")
    ax.set_ylabel("leakage")
    ax.set_title("The v1 leakage metric rewards silence; the normalised one does not",
                 fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{out}/fig3_leakage_metric_{tag}.png", dpi=180)
    plt.close(fig)


def fig_collateral(steer: pd.DataFrame, out: str, tag: str) -> None:
    """Fig 4 - surgical precision: effect against collateral damage on the wanted class."""
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    s = steer.copy()
    s["family"] = s["arm"].map(arm_family)
    for fam, style in ARM_STYLE.items():
        sub = s[s["family"] == fam]
        if sub.empty:
            continue
        agg = sub.groupby("alpha").agg(effect=("effect", "mean"),
                                       collat=("collateral", "mean")).reset_index()
        ax.plot(agg["collat"], agg["effect"], color=style["color"], marker=style["marker"],
                label=style["label"], lw=2, ms=5, alpha=0.9)
    ax.set_xlabel("Collateral change on wanted-class prompts")
    ax.set_ylabel("Reduction in unwanted behaviour")
    ax.set_title(f"Surgical precision - {tag}", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/fig4_collateral_{tag}.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main_table(steer: pd.DataFrame) -> pd.DataFrame:
    """Effect at matched perplexity budgets, mean +/- 95% CI over seeds."""
    rows = []
    for arm, g in steer.groupby("arm"):
        row = {"arm": arm, "family": arm_family(arm)}
        for b in PPL_BUDGETS:
            per_seed = [effect_at_budget(gs, b) for _, gs in g.groupby("seed")]
            per_seed = np.array(per_seed, dtype=float)
            row[f"effect@{b:.0f}%"] = float(np.nanmean(per_seed))
            row[f"ci@{b:.0f}%"] = ci95(per_seed)
        row["max_effect"] = float(g.groupby("alpha")["effect"].mean().max())
        row["min_collateral_at_max"] = float(
            g.loc[g.groupby("alpha")["effect"].mean().idxmax() == g["alpha"], "collateral"].mean()
            if len(g) else np.nan)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(f"effect@{PPL_BUDGETS[1]:.0f}%", ascending=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--tag", required=True, help="e.g. sentiment_gpt2_L8")
    a = ap.parse_args()
    os.makedirs(a.figures, exist_ok=True)

    steer = pd.read_csv(f"{a.results}/steering_{a.tag}.csv")
    metrics = pd.read_csv(f"{a.results}/sae_metrics_{a.tag}.csv")

    fig_tradeoff(steer, a.figures, a.tag)
    fig_buffer_sweep(metrics, a.figures, a.tag)
    fig_metric_gameable(metrics, a.figures, a.tag)
    fig_collateral(steer, a.figures, a.tag)

    tbl = main_table(steer)
    tbl.to_csv(f"{a.results}/main_table_{a.tag}.csv", index=False)

    print(f"\n=== Effect at matched perplexity budget - {a.tag} ===")
    cols = ["arm"] + [f"effect@{b:.0f}%" for b in PPL_BUDGETS] + ["max_effect"]
    print(tbl[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print(f"\n=== SAE quality (matched L0) - {a.tag} ===")
    mcols = [c for c in ["arm", "kind", "tag_frac", "fvu", "l0",
                         "norm_leakage", "raw_leakage", "buffer_ratio"] if c in metrics]
    print(metrics.groupby(["arm"], as_index=False)[
        [c for c in mcols if c not in ("arm", "kind")]].mean().to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nfigures -> {a.figures}/fig*_{a.tag}.png")


if __name__ == "__main__":
    main()
