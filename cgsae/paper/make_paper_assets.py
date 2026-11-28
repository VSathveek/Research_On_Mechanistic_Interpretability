"""
Build every data-derived asset of the paper from cgsae/results/.

Outputs
-------
    figures/*.pdf          vector figures (Times, sized for a 6.3 in text block)
    generated/numbers.tex  one \\newcommand per number quoted in the prose
    generated/tab_*.tex    booktabs tables, \\input by main.tex

Nothing quoted in main.tex is typed by hand: prose numbers are macros from numbers.tex,
tables are \\input from generated/. Re-running this script after new results regenerates
every figure, table and in-text number consistently.

Statistics
----------
Paired two-sided t-tests over seeds, on the effect interpolated at a matched perplexity
budget (upper envelope of each arm's strength sweep). Direction baselines are deterministic
given the fixed data split, so a "paired" test against them is a one-sample test of the SAE
arm's seed variation around a constant; this is stated in the paper. Holm-Bonferroni
adjustment is reported per test family.

Usage:  python make_paper_assets.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "figures")
GEN = os.path.join(HERE, "generated")
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from analyse import effect_at_budget  # noqa: E402

os.makedirs(FIG, exist_ok=True)
os.makedirs(GEN, exist_ok=True)

BUDGET = 10.0
TAUS = ["0.05", "0.10", "0.20", "0.50"]

# Fixed entity -> colour mapping (validated categorical order; neutral grey for nulls).
C_CG, C_VAN, C_SCAR, C_DIM, C_PROBE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
C_NULL, C_INK, C_MUTED = "#8a8984", "#0b0b0b", "#52514e"
BLUE_SEQ = ["#b8cff0", "#5a97e0", "#1c4f94"]          # light -> dark, one hue

RUNS = [
    dict(key="R1", tag="sentiment_gpt2_L7", axis="Sentiment", model="GPT-2", layer=7,
         primary=True),
    dict(key="R2", tag="toxicity_gpt2_L8", axis="Toxicity", model="GPT-2", layer=8,
         primary=True),
    dict(key="R3", tag="sentiment_Qwen2.5-1.5B-Instruct_L15", axis="Sentiment",
         model="Qwen2.5-1.5B", layer=15, primary=True),
    dict(key="R4", tag="sentiment_Qwen2.5-1.5B-Instruct_L18", axis="Sentiment",
         model="Qwen2.5-1.5B", layer=18, primary=True),
    dict(key="R5", tag="toxicity_Qwen2.5-1.5B-Instruct_L10", axis="Toxicity",
         model="Qwen2.5-1.5B", layer=10, primary=True),
    dict(key="R6", tag="toxicity_Qwen2.5-1.5B-Instruct_L15", axis="Toxicity",
         model="Qwen2.5-1.5B", layer=15, primary=False),
]
for r in RUNS:
    r["label"] = f"{r['axis']}, {r['model']}, L{r['layer']}"
    r["short"] = f"{r['axis'][:4]}., {'GPT-2' if r['model']=='GPT-2' else 'Qwen'} L{r['layer']}"
PRIMARY = [r for r in RUNS if r["primary"]]

NUM: dict[str, str] = {}


def num(name: str, value) -> None:
    """Register a number for numbers.tex (macro names must be letters only)."""
    assert name.isalpha(), name
    NUM[name] = str(value)


def fmt_p(p: float) -> str:
    if p != p:
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def star(p: float) -> str:
    return "$^{*}$" if (p == p and p < 0.05) else ""


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs passed through)."""
    idx = [i for i, p in enumerate(pvals) if p == p]
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    adj = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------
_steer: dict[str, pd.DataFrame] = {}


def steer(tag: str) -> pd.DataFrame:
    if tag not in _steer:
        _steer[tag] = pd.read_csv(os.path.join(RES, f"steering_{tag}.csv"))
    return _steer[tag]


def per_seed(tag: str, arm: str, budget: float = BUDGET) -> np.ndarray:
    s = steer(tag)
    g = s[s.arm == arm]
    if g.empty:
        return np.array([])
    return np.array([effect_at_budget(gs, budget) for _, gs in g.groupby("seed")], float)


def paired(tag: str, a: str, b: str, budget: float = BUDGET) -> dict:
    xa, xb = per_seed(tag, a, budget), per_seed(tag, b, budget)
    out = dict(ma=np.nanmean(xa) if len(xa) else np.nan,
               mb=np.nanmean(xb) if len(xb) else np.nan, nseed=0, p=np.nan, d=np.nan,
               lo=np.nan, hi=np.nan, sa=np.nan, sb=np.nan)
    if len(xa) == 0 or len(xb) == 0:
        return out
    if len(xb) == 1 or np.allclose(xb, xb[0]):          # deterministic baseline
        xb = np.full_like(xa, np.nanmean(xb))
    n = min(len(xa), len(xb))
    d = (xa[:n] - xb[:n])
    d = d[~np.isnan(d)]
    out.update(nseed=len(d), d=float(np.mean(d)) if len(d) else np.nan,
               sa=float(np.nanstd(xa, ddof=1)) if len(xa) > 1 else np.nan,
               sb=float(np.nanstd(xb, ddof=1)) if len(xb) > 1 else np.nan)
    if len(d) >= 2 and np.std(d, ddof=1) > 0:
        se = np.std(d, ddof=1) / np.sqrt(len(d))
        t = np.mean(d) / se
        out["p"] = float(2 * (1 - stats.t.cdf(abs(t), len(d) - 1)))
        h = stats.t.ppf(0.975, len(d) - 1) * se
        out["lo"], out["hi"] = float(np.mean(d) - h), float(np.mean(d) + h)
    return out


def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def config(tag: str) -> dict:
    p = os.path.join(RES, f"config_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ---------------------------------------------------------------------------
# matplotlib house style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "axes.labelcolor": C_INK, "text.color": C_INK,
    "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name: str) -> None:
    fig.savefig(os.path.join(FIG, name))
    fig.savefig(os.path.join(FIG, name.replace(".pdf", ".png")), dpi=200)
    plt.close(fig)


# ===========================================================================
# 1. Synthetic testbed (Experiment 1)
# ===========================================================================
def exp1() -> None:
    s = pd.read_csv(os.path.join(RES, "exp1_synthetic_summary.csv"))
    raw = pd.read_csv(os.path.join(RES, "exp1_synthetic_raw.csv"))
    n_seeds = int(raw.seed.nunique())
    num("nSynthSeeds", n_seeds)

    def br(t):
        nt = max(1, int(32 * t))
        return (32 - 2 * nt) / (2 * nt)

    # per-seed specificity so we can put an interval on the ratio
    rows = []
    for (lam, tf), g in raw.groupby(["lambda", "tag_frac"]):
        b = g[g.ablation == "B-block"].set_index("seed").d_auc
        r = g[g.ablation == "random"].set_index("seed").d_auc
        rows.append(dict(lam=lam, tau=tf, buf=br(tf),
                         b=b.mean(), b_ci=ci95(b.values), r=r.mean(), r_ci=ci95(r.values),
                         spec=b.mean() / r.mean()))
    sp = pd.DataFrame(rows).sort_values(["lam", "tau"])
    sp.to_csv(os.path.join(GEN, "exp1_specificity_computed.csv"), index=False)

    on = sp[sp.lam == 1.0]
    num("specMax", f"{on.spec.max():.1f}")
    num("specMaxBuf", f"{on.loc[on.spec.idxmax(), 'buf']:.2f}")
    num("specNoBufLamOne", f"{on[on.tau == 0.5].spec.iloc[0]:.1f}")
    num("specNoBufLamFive", f"{sp[(sp.lam == 5.0) & (sp.tau == 0.5)].spec.iloc[0]:.2f}")
    num("specMaxLamFive", f"{sp[sp.lam == 5.0].spec.max():.1f}")
    off = sp[sp.lam == 0.0]
    num("specOffMin", f"{off.spec.min():.2f}")
    num("specOffMax", f"{off.spec.max():.2f}")

    # table: specificity
    lines = [r"\begin{tabular}{@{}cc ccc ccc@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{$\Delta$AUC, $\lambda=1$} & "
             r"\multicolumn{3}{c}{Specificity $S$} \\",
             r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
             r"$\tau$ & Buffer ratio & B-block & Random block & & "
             r"$\lambda=0$ & $\lambda=1$ & $\lambda=5$ \\", r"\midrule"]
    for tf in sorted(sp.tau.unique()):
        a0 = sp[(sp.lam == 0) & (sp.tau == tf)].iloc[0]
        a1 = sp[(sp.lam == 1) & (sp.tau == tf)].iloc[0]
        a5 = sp[(sp.lam == 5) & (sp.tau == tf)].iloc[0]
        lines.append(
            f"{tf:.2f} & {a1.buf:.2f} & ${a1.b:+.3f}\\pm{a1.b_ci:.3f}$ & "
            f"${a1.r:+.3f}\\pm{a1.r_ci:.3f}$ & & {a0.spec:.2f} & "
            f"\\textbf{{{a1.spec:.1f}}} & {a5.spec:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_exp1_specificity.tex"), "w").write("\n".join(lines))

    # dissociation at lambda=1, tau=0.20
    d = s[(s["lambda"] == 1.0) & np.isclose(s.tag_frac, 0.20)].set_index("ablation")
    for key, nm in [("B-block", "B"), ("G-block", "G"), ("random", "R")]:
        num(f"diss{nm}Unw", f"{d.loc[key, 'd_unwanted']:+.3f}")
        num(f"diss{nm}Wan", f"{d.loc[key, 'd_wanted']:+.3f}")
    # per-seed CIs for the dissociation table
    sub = raw[(raw["lambda"] == 1.0) & np.isclose(raw.tag_frac, 0.20)]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Ablated block & $\Delta$acc.\ unwanted class & $\Delta$acc.\ wanted class & "
             r"$\Delta$AUC \\", r"\midrule"]
    for key, nm in [("B-block", "B-block (unwanted side)"),
                    ("G-block", "G-block (wanted side)"),
                    ("random", "Random block, equal width")]:
        g = sub[sub.ablation == key]
        lines.append(f"{nm} & ${g.d_acc_unwanted.mean():+.3f}\\pm{ci95(g.d_acc_unwanted.values):.3f}$"
                     f" & ${g.d_acc_wanted.mean():+.3f}\\pm{ci95(g.d_acc_wanted.values):.3f}$"
                     f" & ${g.d_auc.mean():+.3f}\\pm{ci95(g.d_auc.values):.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_exp1_dissociation.tex"), "w").write("\n".join(lines))
    num("synthRouting", f"{raw[(raw['lambda'] == 1.0)].routing.mean():.2f}")
    num("synthRoutingOff", f"{raw[(raw['lambda'] == 0.0)].routing.mean():.3f}")
    num("synthBAliveMin", f"{100 * raw[raw['lambda'] > 0].b_alive.min():.0f}")

    # ---- figure: specificity vs buffer + dissociation ----
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    lam_col = {0.0: C_NULL, 1.0: BLUE_SEQ[1], 5.0: BLUE_SEQ[2]}
    lam_mk = {0.0: "x", 1.0: "o", 5.0: "s"}
    for lam in [0.0, 1.0, 5.0]:
        g = sp[sp.lam == lam].sort_values("tau")
        ax.plot(g.tau, g.spec, marker=lam_mk[lam], color=lam_col[lam],
                label=("penalty off ($\\lambda=0$)" if lam == 0 else f"$\\lambda={lam:g}$"))
    ax.axhline(1.0, color=C_MUTED, ls="--", lw=0.8)
    ax.text(0.10, 0.93, "no specificity ($S=1$)", ha="left", va="top", fontsize=7, color=C_MUTED)
    ax.set_ylim(0.7, 25)
    ax.set_yscale("log")
    ax.set_xlabel(r"tag fraction $\tau$  ($\tau=0.5$: no buffer)")
    ax.set_ylabel(r"specificity $S$ (log scale)")
    ax.set_xticks(sorted(sp.tau.unique()))
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(sorted(sp.tau.unique()))
    sec.set_xticklabels([f"{br(t):.2f}" for t in sorted(sp.tau.unique())], fontsize=7)
    sec.set_xlabel("buffer ratio", fontsize=7.5, color=C_MUTED)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(a) Buffer width governs ablation specificity", loc="left", fontsize=8.5)

    ax = axes[1]
    order = [("B-block", "B-block"), ("G-block", "G-block"), ("random", "random\n(equal width)")]
    x = np.arange(3)
    w = 0.36
    unw = [d.loc[k, "d_unwanted"] for k, _ in order]
    wan = [d.loc[k, "d_wanted"] for k, _ in order]
    ax.bar(x - w / 2 - 0.01, unw, w, color=C_VAN, label="unwanted-class acc.")
    ax.bar(x + w / 2 + 0.01, wan, w, color=C_CG, label="wanted-class acc.")
    for xi, (u, v) in enumerate(zip(unw, wan)):
        ax.text(xi - w / 2, u + (0.012 if u >= 0 else -0.012), f"{u:+.2f}", ha="center",
                va="bottom" if u >= 0 else "top", fontsize=6.8, color=C_INK)
        ax.text(xi + w / 2, v + (0.012 if v >= 0 else -0.012), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=6.8, color=C_INK)
    ax.axhline(0, color=C_MUTED, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in order])
    ax.set_ylabel(r"$\Delta$ accuracy after ablation")
    ax.set_ylim(-0.6, 0.55)
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    ax.set_title(r"(b) Double dissociation ($\lambda=1,\ \tau=0.20$)", loc="left", fontsize=8.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_synthetic.pdf")


# ===========================================================================
# 2. Penalty forms and the leakage metric
# ===========================================================================
def penalty_forms() -> None:
    m = pd.read_csv(os.path.join(RES, "penform_metrics_sentiment_gpt2_L8.csv"))
    cg = m[m.kind == "cgsae"].copy()
    agg = (cg.groupby(["pen_mode", "lambda_cg"])
             .agg(fvu=("fvu", "mean"), leak=("norm_leakage", "mean"),
                  routing=("routing", "mean"), alive=("b_alive_frac", "mean"),
                  dead=("n_dead_latents", "mean"), n=("seed", "nunique"))
             .reset_index())
    van_fvu = m[m.kind == "vanilla"].fvu.mean()
    r = np.corrcoef(cg.norm_leakage, cg.routing)[0, 1]
    num("leakRoutingCorr", f"{r:+.2f}")
    num("penformSeeds", int(cg.seed.nunique()))
    num("penformConfigs", len(agg))
    best_leak = agg.loc[agg.leak.idxmin()]
    best_rout = agg.loc[agg.routing.idxmax()]
    num("bestLeakValue", f"{best_leak.leak:.4f}")
    num("bestLeakDeadPct", f"{100 * (1 - best_leak.alive):.0f}")
    num("bestLeakRouting", f"{best_leak.routing:.4f}")
    num("bestLeakMode", best_leak.pen_mode)
    num("bestLeakLambda", f"{best_leak.lambda_cg:g}")
    num("bestRoutValue", f"{best_rout.routing:.2f}")
    num("bestRoutLeak", f"{best_rout.leak:.3f}")
    num("bestRoutAlivePct", f"{100 * best_rout.alive:.0f}")
    num("leakRatioBestVsRout", f"{best_rout.leak / best_leak.leak:.0f}")
    sq = agg[agg.pen_mode == "sq_mean"]
    num("sqRoutMax", f"{sq.routing.max():.3f}")
    num("vanFVUpenform", f"{van_fvu:.3f}")
    ct = agg[agg.pen_mode == "contrast"].set_index("lambda_cg")
    num("contrastFVUlamOne", f"{ct.loc[1.0, 'fvu']:.3f}")
    num("contrastRoutLamOne", f"{ct.loc[1.0, 'routing']:.2f}")
    num("contrastFVUlamTen", f"{ct.loc[10.0, 'fvu']:.2f}")

    names = {"sq_mean": r"squared mean (v1)", "mean": "mean", "ratio": "ratio",
             "contrast": r"\textbf{contrastive (ours)}"}
    lines = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"Objective & $\lambda$ & FVU $\downarrow$ & Leakage $\downarrow$ & "
             r"Routing $\uparrow$ & B-block alive & Dead latents \\", r"\midrule"]
    for mode in ["sq_mean", "mean", "ratio", "contrast"]:
        g = agg[agg.pen_mode == mode].sort_values("lambda_cg")
        for i, (_, rr) in enumerate(g.iterrows()):
            lab = names[mode] if i == 0 else ""
            leak = f"{rr.leak:.4f}"
            if rr.pen_mode == best_leak.pen_mode and rr.lambda_cg == best_leak.lambda_cg:
                leak = r"\underline{" + leak + "}"
            rout = f"{rr.routing:.3f}"
            if rr.pen_mode == best_rout.pen_mode and rr.lambda_cg == best_rout.lambda_cg:
                rout = r"\textbf{" + rout + "}"
            lines.append(f"{lab} & {rr.lambda_cg:g} & {rr.fvu:.3f} & {leak} & {rout} & "
                         f"{100 * rr.alive:.0f}\\% & {rr.dead:.0f} \\\\")
        if mode != "contrast":
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\midrule",
              f"Vanilla TopK SAE & -- & {van_fvu:.3f} & -- & -- & -- & "
              f"{m[m.kind == 'vanilla'].n_dead_latents.mean():.0f} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_penform.tex"), "w").write("\n".join(lines))

    # calibration sweep of the v1 objective (lambda up to 30)
    c = pd.read_csv(os.path.join(RES, "calib_sqmean_sentiment_gpt2_L8.csv"))
    c = c[c.kind == "cgsae"].sort_values("lambda_cg")
    num("sqLeakLamZero", f"{c.iloc[0].norm_leakage:.3f}")
    num("sqLeakLamThirty", f"{c.iloc[-1].norm_leakage:.3f}")
    num("sqFVUspread", f"{c.fvu.max() - c.fvu.min():.1e}".replace("e-0", r"\times10^{-") + "}")
    pen30 = 30 * c.iloc[-1].raw_leakage ** 2
    num("sqPenTermThirty", f"{pen30:.1e}".replace("e-0", r"\times10^{-") + "}")
    num("sqPenShare", f"{100 * pen30 / c.iloc[-1].fvu:.1f}")
    lines = [r"\begin{tabular}{@{}cccc@{}}", r"\toprule",
             r"$\lambda$ & Normalised leakage & Raw leakage & FVU \\", r"\midrule"]
    for _, rr in c.iterrows():
        lines.append(f"{rr.lambda_cg:g} & {rr.norm_leakage:.4f} & {rr.raw_leakage:.4f} & "
                     f"{rr.fvu:.5f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_sqmean_calib.tex"), "w").write("\n".join(lines))

    # rescaling demonstration (exact, analytic): leakage under z -> c z
    rng = np.random.RandomState(0)
    d, n = 128, 400
    z = np.zeros((n, d))
    lab = np.r_[np.ones(n // 2), np.zeros(n // 2)]
    g, b = np.arange(12), np.arange(d - 12, d)
    z[:200, :12] = rng.rand(200, 12)
    z[:200, -12:] = rng.rand(200, 12) * 0.25
    z[200:, -12:] = rng.rand(200, 12)
    z[200:, :12] = rng.rand(200, 12) * 0.25

    def raw_l(zz):
        return 0.5 * (zz[lab == 1][:, b].mean() + zz[lab == 0][:, g].mean())

    def norm_l(zz):
        gm, bm = zz[:, g].sum(1), zz[:, b].sum(1)
        t = gm + bm + 1e-8
        return 0.5 * ((bm[lab == 1] / t[lab == 1]).mean() + (gm[lab == 0] / t[lab == 0]).mean())

    lines = [r"\begin{tabular}{@{}ccc@{}}", r"\toprule",
             r"Scale $c$ ($\mathbf z\mapsto c\,\mathbf z$) & Raw leakage & Normalised leakage \\",
             r"\midrule"]
    for cc in [1.0, 0.1, 0.01]:
        lines.append(f"{cc:g} & {raw_l(z * cc):.5f} & {norm_l(z * cc):.5f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_rescale.tex"), "w").write("\n".join(lines))

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.45))
    style = {"sq_mean": (C_NULL, "x", "squared mean (v1)"), "mean": (C_VAN, "s", "mean"),
             "ratio": (C_SCAR, "^", "ratio"), "contrast": (C_CG, "o", "contrastive (ours)")}
    for mode, (col, mk, lab) in style.items():
        g = agg[agg.pen_mode == mode]
        axes[0].scatter(g.leak, g.routing, color=col, marker=mk, s=34, label=lab,
                        edgecolor="white", linewidth=0.6, zorder=3)
        axes[1].scatter(g.leak, 100 * g.alive, color=col, marker=mk, s=34, label=lab,
                        edgecolor="white", linewidth=0.6, zorder=3)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("normalised leakage (lower = 'better')")
    axes[0].set_ylabel("routing (higher = better)")
    axes[1].set_ylabel("B-block latents alive (%)")
    axes[0].annotate(f"best leakage\nrouting {best_leak.routing:.4f}",
                     xy=(best_leak.leak, best_leak.routing), xytext=(best_leak.leak * 2.2, 0.17),
                     fontsize=6.8, color=C_INK, arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=0.6))
    axes[1].annotate(f"best leakage\n{100 * (1 - best_leak.alive):.0f}% of block dead",
                     xy=(best_leak.leak, 100 * best_leak.alive),
                     xytext=(best_leak.leak * 2.2, 40), fontsize=6.8, color=C_INK,
                     arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=0.6))
    axes[0].set_title(f"(a) Leakage vs routing, $r={r:+.2f}$", loc="left", fontsize=8.5)
    axes[1].set_title("(b) Low leakage is reached by silencing the block", loc="left",
                      fontsize=8.5)
    axes[0].legend(frameon=False, loc="center right", fontsize=6.8)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_penform.pdf")


# ===========================================================================
# 3. SAE quality per run
# ===========================================================================
def sae_quality() -> None:
    lines = [r"\begin{tabular}{@{}ll ccccc@{}}", r"\toprule",
             r"Run & SAE & FVU & L$_0$ & Routing & B-block alive & Dead latents \\",
             r"\midrule"]
    fvu_van, fvu_cg = [], []
    for ri, r in enumerate(RUNS):
        m = pd.read_csv(os.path.join(RES, f"sae_metrics_{r['tag']}.csv"))
        first = True
        for arm in ["vanilla", "scar", "cgsae@0.05", "cgsae@0.10", "cgsae@0.20", "cgsae@0.50"]:
            g = m[m.arm == arm]
            if g.empty:
                continue
            lab = r["short"] if first else ""
            first = False
            name = {"vanilla": "Vanilla", "scar": "Scar"}.get(arm, f"CG-SAE $\\tau={arm[6:]}$")
            rout = f"{g.routing.mean():.3f}" if "routing" in g and g.routing.notna().any() else "--"
            alive = (f"{100 * g.b_alive_frac.mean():.0f}\\%"
                     if "b_alive_frac" in g and g.b_alive_frac.notna().any() else "--")
            lines.append(f"{lab} & {name} & {g.fvu.mean():.3f}$\\pm${g.fvu.std(ddof=1):.3f} & "
                         f"{g.l0.mean():.0f} & {rout} & {alive} & {g.n_dead_latents.mean():.0f} \\\\")
            if arm == "vanilla":
                fvu_van.append((r["model"], g.fvu.mean()))
            if arm == "cgsae@0.10":
                fvu_cg.append((r["model"], g.fvu.mean()))
        if ri < len(RUNS) - 1:
            lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_sae_quality.tex"), "w").write("\n".join(lines))
    gv = [v for mdl, v in fvu_van if mdl == "GPT-2"]
    qv = [v for mdl, v in fvu_van if mdl != "GPT-2"]
    gc = [v for mdl, v in fvu_cg if mdl == "GPT-2"]
    qc = [v for mdl, v in fvu_cg if mdl != "GPT-2"]
    num("fvuVanGPT", f"{np.mean(gv):.3f}")
    num("fvuVanQwen", f"{np.mean(qv):.3f}")
    num("fvuCGGPT", f"{np.mean(gc):.3f}")
    num("fvuCGQwen", f"{np.mean(qc):.3f}")
    num("evVanGPT", f"{100 * (1 - np.mean(gv)):.0f}")
    num("evVanQwen", f"{100 * (1 - np.mean(qv)):.0f}")


# ===========================================================================
# 4. Block localisation (reserved block vs size-matched random block)
# ===========================================================================
def block() -> None:
    res = []
    for r in RUNS:
        for tf in TAUS:
            o = paired(r["tag"], f"cgsae@{tf}/block", f"cgsae@{tf}/blockrand")
            res.append(dict(run=r["key"], primary=r["primary"], tau=tf, **o))
    df = pd.DataFrame(res)
    prim = df[df.primary].copy()
    prim["p_holm"] = holm(prim.p.tolist())
    df = df.merge(prim[["run", "tau", "p_holm"]], on=["run", "tau"], how="left")
    df.to_csv(os.path.join(GEN, "block_tests.csv"), index=False)

    num("blockSig", int((prim.p < 0.05).sum()))
    num("blockTotal", len(prim))
    num("blockSigHolm", int((prim.p_holm < 0.05).sum()))
    num("blockRuns", prim.run.nunique())
    for tf, nm in [("0.05", "Five"), ("0.10", "Ten"), ("0.20", "Twenty"), ("0.50", "Fifty")]:
        num(f"blockSigTau{nm}", int((prim[prim.tau == tf].p < 0.05).sum()))
    num("blockPositive", int((prim.d > 0).sum()))
    # Individual cells have 3-5 seeds and are underpowered; the direction is the evidence.
    # Exact one-sided sign test on the 20 primary cells (cells share evaluation prompts, so
    # this treats them as independent and should be read as descriptive, not confirmatory).
    k, nn = int((prim.d > 0).sum()), int(prim.d.notna().sum())
    num("blockSignP", f"{stats.binomtest(k, nn, 0.5, alternative='greater').pvalue:.1e}"
        .replace("e-0", r"\times10^{-") + "}")
    q = df[(~df.primary)]
    num("blockOffPositive", int((q.d > 0).sum()))

    lines = [r"\begin{tabular}{@{}l cccc@{}}", r"\toprule",
             r"Run & $\tau=0.05$ & $\tau=0.10$ & $\tau=0.20$ & $\tau=0.50$ (no buffer) \\",
             r"\midrule"]
    for r in RUNS:
        if not r["primary"]:
            lines.append(r"\midrule")
        cells = []
        for tf in TAUS:
            o = df[(df.run == r["key"]) & (df.tau == tf)].iloc[0]
            cells.append(f"${o.d:+.3f}$\\,/\\,{fmt_p(o.p)}{star(o.p)}")
        lab = r["label"] + ("$^{\\dagger}$" if not r["primary"] else "")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_block.tex"), "w").write("\n".join(lines))

    # figure: small multiples, block vs random block per run
    fig, axes = plt.subplots(1, 5, figsize=(6.3, 2.05), sharey=True)
    for ax, r in zip(axes, PRIMARY):
        x = np.arange(len(TAUS))
        mb, mr, eb, er, ps = [], [], [], [], []
        for tf in TAUS:
            xb = per_seed(r["tag"], f"cgsae@{tf}/block")
            xr = per_seed(r["tag"], f"cgsae@{tf}/blockrand")
            mb.append(np.nanmean(xb)); mr.append(np.nanmean(xr))
            eb.append(ci95(xb)); er.append(ci95(xr))
            ps.append(df[(df.run == r["key"]) & (df.tau == tf)].p.iloc[0])
        w = 0.36
        ax.bar(x - w / 2 - 0.01, mb, w, yerr=eb, color=C_CG, error_kw=dict(lw=0.6, capsize=1.5),
               label="reserved B-block")
        ax.bar(x + w / 2 + 0.01, mr, w, yerr=er, color=C_NULL,
               error_kw=dict(lw=0.6, capsize=1.5), label="random block, equal width")
        for xi, p, top in zip(x, ps, np.array(mb) + np.array(eb)):
            if p == p and p < 0.05:
                ax.text(xi - w / 2, top + 0.008, "*", ha="center", va="bottom", fontsize=9,
                        color=C_INK)
        ax.set_xticks(x)
        ax.set_xticklabels([t[1:] if t.startswith("0") else t for t in TAUS], fontsize=6.8)
        ax.set_title(f"{r['axis']}\n{'GPT-2' if r['model'] == 'GPT-2' else 'Qwen'} L{r['layer']}",
                     fontsize=7.5)
        ax.axhline(0, color=C_MUTED, lw=0.6)
    axes[0].set_ylabel(f"effect at {BUDGET:.0f}% perplexity")
    fig.supxlabel(r"tag fraction $\tau$", fontsize=8, y=0.06)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig_block.pdf")


# ===========================================================================
# 5. Comparisons: CG-SAE vs SAE-SSV / Scar, and SAE vs difference-in-means
# ===========================================================================
def comparisons() -> None:
    rows = []
    for r in RUNS:
        for n in [64, 256, 1024]:
            for base, bname in [("vanilla", "SAE-SSV"), ("scar", "Scar")]:
                o = paired(r["tag"], f"cgsae@0.10/free@{n}", f"{base}/free@{n}")
                rows.append(dict(run=r["key"], model=r["model"], primary=r["primary"], n=n,
                                 base=bname, **o))
    df = pd.DataFrame(rows)
    fam = df[(df.base == "SAE-SSV") & (df.n.isin([256, 1024]))].copy()
    fam["p_holm"] = holm(fam.p.tolist())
    df = df.merge(fam[["run", "n", "base", "p_holm"]], on=["run", "n", "base"], how="left")
    df.to_csv(os.path.join(GEN, "comparison_tests.csv"), index=False)

    g = fam[fam.model == "GPT-2"]
    q = fam[fam.model != "GPT-2"]
    num("vsSSVgptSig", int((g.p < 0.05).sum()))
    num("vsSSVgptTotal", len(g))
    num("vsSSVgptPmin", f"{g.p.min():.2f}")
    num("vsSSVgptPmax", f"{g.p.max():.2f}")
    num("vsSSVqwenSig", int((q.p < 0.05).sum()))
    num("vsSSVqwenTotal", len(q))
    num("vsSSVqwenPositive", int((q.d > 0).sum()))
    num("vsSSVqwenSigHolm", int((q.p_holm < 0.05).sum()))
    num("vsSSVgptPositive", int((g.d > 0).sum()))
    qp = q[q.primary]
    num("vsSSVqwenPrimSig", int((qp.p < 0.05).sum()))
    num("vsSSVqwenPrimTotal", len(qp))
    sc = df[(df.base == "Scar") & (df.n.isin([256, 1024]))]
    num("vsScarSig", int((sc.p < 0.05).sum()))
    num("vsScarTotal", len(sc))
    num("vsScarQwenSig", int((sc[sc.model != "GPT-2"].p < 0.05).sum()))

    lines = [r"\begin{tabular}{@{}l c c cc cc@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{SAE-SSV} & \multicolumn{2}{c}{Scar} \\",
             r"\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             r"Run & $n$ & CG-SAE & effect & $p$ & effect & $p$ \\", r"\midrule"]
    for ri, r in enumerate(RUNS):
        for j, n in enumerate([256, 1024]):
            a = df[(df.run == r["key"]) & (df.n == n) & (df.base == "SAE-SSV")].iloc[0]
            b = df[(df.run == r["key"]) & (df.n == n) & (df.base == "Scar")].iloc[0]
            lab = (r["label"] + ("$^{\\dagger}$" if not r["primary"] else "")) if j == 0 else ""
            lines.append(f"{lab} & {n} & ${a.ma:+.3f}$ & ${a.mb:+.3f}$ & {fmt_p(a.p)}{star(a.p)} & "
                         f"${b.mb:+.3f}$ & {fmt_p(b.p)}{star(b.p)} \\\\")
        if ri == 1:
            lines.append(r"\midrule")
        elif ri < len(RUNS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_comparisons.tex"), "w").write("\n".join(lines))

    # --- vs difference-in-means: pre-specified arms, no post-hoc 'best arm' selection ---
    arms = [("cgsae@0.10/free@1024", "CG-SAE"), ("vanilla/free@1024", "SAE-SSV"),
            ("scar/free@1024", "Scar"), ("cgsae@0.10/block", "CG-SAE block")]
    drows = []
    for r in RUNS:
        for arm, nm in arms:
            o = paired(r["tag"], arm, "diff_in_means")
            drows.append(dict(run=r["key"], axis=r["axis"], model=r["model"],
                              primary=r["primary"], arm=nm, **o))
    dd = pd.DataFrame(drows)
    dd.to_csv(os.path.join(GEN, "dim_tests.csv"), index=False)
    lines = [r"\begin{tabular}{@{}l c cccc@{}}", r"\toprule",
             r"Run & Diff.-in-means & CG-SAE & SAE-SSV & Scar & CG-SAE block \\", r"\midrule"]
    for r in RUNS:
        sub = dd[dd.run == r["key"]].set_index("arm")
        dim = sub.iloc[0].mb
        cells = []
        for _, nm in arms:
            o = sub.loc[nm]
            better = o.ma > dim
            v = f"{o.ma:+.3f}"
            if o.p == o.p and o.p < 0.05:
                v = (r"\textbf{" + v + "}") if better else (r"\textit{" + v + "}")
            cells.append(v + star(o.p))
        lab = r["label"] + ("$^{\\dagger}$" if not r["primary"] else "")
        lines.append(f"{lab} & ${dim:+.3f}$ & " + " & ".join(cells) + r" \\")
        if r["key"] == "R2":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_dim.tex"), "w").write("\n".join(lines))

    def cnt(axis, better):
        x = dd[(dd.axis == axis) & dd.primary & (dd.p < 0.05)]
        return int((x.ma > x.mb).sum() if better else (x.ma < x.mb).sum())
    num("dimSentSAEworse", cnt("Sentiment", False))
    num("dimSentSAEbetter", cnt("Sentiment", True))
    num("dimToxSAEbetter", cnt("Toxicity", True))
    num("dimToxSAEworse", cnt("Toxicity", False))
    num("dimSentTotal", int((dd.axis.eq("Sentiment") & dd.primary).sum()))
    num("dimToxTotal", int((dd.axis.eq("Toxicity") & dd.primary).sum()))
    for key, nm in [("R2", "GptTox"), ("R5", "QwenTox"), ("R1", "GptSent"), ("R3", "QwenSent")]:
        sub = dd[(dd.run == key)].set_index("arm")
        num(f"dim{nm}", f"{sub.iloc[0].mb:+.3f}")
        num(f"scar{nm}", f"{sub.loc['Scar'].ma:+.3f}")
        num(f"scarP{nm}", fmt_p(sub.loc["Scar"].p))
        num(f"cg{nm}", f"{sub.loc['CG-SAE'].ma:+.3f}")
        num(f"cgP{nm}", fmt_p(sub.loc["CG-SAE"].p))

    # figure: dumbbell, CG-SAE vs SAE-SSV at n=1024 and n=256
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4), sharey=True)
    order = [r for r in RUNS]
    ylab = [r["label"] + (" †" if not r["primary"] else "") for r in order]
    for ax, n in zip(axes, [256, 1024]):
        for yi, r in enumerate(order):
            a = df[(df.run == r["key"]) & (df.n == n) & (df.base == "SAE-SSV")].iloc[0]
            y = len(order) - 1 - yi
            ax.plot([a.mb, a.ma], [y, y], color="#cfcec9", lw=2.0, zorder=1)
            ax.scatter([a.mb], [y], color=C_VAN, s=30, zorder=3, marker="s",
                       label="SAE-SSV" if yi == 0 else None, edgecolor="white", linewidth=0.5)
            ax.scatter([a.ma], [y], color=C_CG, s=30, zorder=3, marker="o",
                       label="CG-SAE ($\\tau=0.10$)" if yi == 0 else None,
                       edgecolor="white", linewidth=0.5)
            ax.text(max(a.ma, a.mb) + 0.012, y, ('p<0.001' if a.p < 0.001 else f'p={a.p:.3f}'),
                    va="center", fontsize=6.6,
                    color=C_INK if (a.p == a.p and a.p < 0.05) else C_MUTED)
        ax.axhline(len(order) - 2.5, color=C_MUTED, lw=0.6, ls=":")
        ax.set_title(f"$n={n}$ latents intervened on", loc="left", fontsize=8.5)
        ax.set_xlabel(f"effect at {BUDGET:.0f}% perplexity")
        ax.set_xlim(-0.02, 0.36)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(ylab[::-1], fontsize=7)
    axes[0].text(0.345, len(order) - 1.5, "GPT-2", ha="right", va="center", fontsize=7,
                 color=C_MUTED)
    axes[0].text(0.345, len(order) - 3.2, "Qwen2.5-1.5B", ha="right", va="center",
                 fontsize=7, color=C_MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.6, -0.04))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.06, 1, 1))
    save(fig, "fig_scale.pdf")


# ===========================================================================
# 6. Trade-off curves
# ===========================================================================
def tradeoff() -> None:
    panel_runs = [RUNS[0], RUNS[1], RUNS[2], RUNS[4]]
    series = [("cgsae@0.10/free@1024", "CG-SAE ($\\tau=0.10$)", C_CG, "o"),
              ("vanilla/free@1024", "SAE-SSV", C_VAN, "s"),
              ("scar/free@1024", "Scar", C_SCAR, "^"),
              ("diff_in_means", "difference-in-means", C_DIM, "D"),
              ("random_dir", "random direction", C_NULL, "x")]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.3))
    for ax, r in zip(axes.flat, panel_runs):
        s = steer(r["tag"])
        for arm, lab, col, mk in series:
            g = s[s.arm == arm]
            if g.empty:
                continue
            a = g.groupby("alpha").agg(e=("effect", "mean"), p=("ppl_pct", "mean")).reset_index()
            ci = g.groupby("alpha").effect.apply(lambda v: ci95(v.values)).values
            a = a.assign(ci=ci).sort_values("p")
            a = a[a.p <= 60]
            ax.plot(a.p, a.e, color=col, marker=mk, label=lab, ms=3.8, lw=1.4)
            if ci.max() > 0:
                ax.fill_between(a.p, a.e - a.ci, a.e + a.ci, color=col, alpha=0.12, lw=0)
        ax.axvline(BUDGET, color=C_MUTED, ls=":", lw=0.8)
        ax.axhline(0, color=C_MUTED, lw=0.6)
        ax.set_title(r["label"], loc="left", fontsize=8.5)
        ax.set_xlim(-1, 60)
    for ax in axes[1]:
        ax.set_xlabel("perplexity increase on held-out WikiText (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("reduction in unwanted score")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.2, w_pad=1.2)
    save(fig, "fig_tradeoff.pdf")


# ===========================================================================
# 7. Layer scans
# ===========================================================================
def layers() -> None:
    scans = [("sentiment_gpt2", "Sentiment, GPT-2"), ("toxicity_gpt2", "Toxicity, GPT-2"),
             ("sentiment_Qwen2.5-1.5B-Instruct", "Sentiment, Qwen2.5-1.5B"),
             ("toxicity_Qwen2.5-1.5B-Instruct", "Toxicity, Qwen2.5-1.5B")]
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.1), sharex="col")
    for j, (tag, lab) in enumerate(scans):
        d = pd.read_csv(os.path.join(RES, f"layer_scan_{tag}.csv"))
        lp = int(d.loc[d.probe_auc.idxmax(), "layer"])
        lsteer = int(d.loc[d.efficiency.idxmax(), "layer"])
        e_at_probe = float(d[d.layer == lp].efficiency.iloc[0])
        e_max = float(d.efficiency.max())
        rows.append(dict(scan=lab, n=len(d), lp=lp, auc=float(d.probe_auc.max()), lsteer=lsteer,
                         emax=e_max, eff_at=float(d[d.layer == lsteer].effect.iloc[0]),
                         ppl_at=float(d[d.layer == lsteer].ppl_pct.iloc[0]),
                         frac=e_at_probe / e_max))
        ax = axes[0, j]
        ax.plot(d.layer, d.probe_auc, color=C_CG, marker="o", ms=2.6, lw=1.2)
        ax.axvline(lp, color=C_MUTED, ls="--", lw=0.8)
        ax.set_title(lab.replace(", ", "\n"), fontsize=7.5)
        ax2 = axes[1, j]
        ax2.plot(d.layer, d.efficiency, color=C_CG, marker="o", ms=2.6, lw=1.2)
        ax2.axvline(lp, color=C_MUTED, ls="--", lw=0.8)
        ax2.axvline(lsteer, color=C_INK, ls="-", lw=0.8)
        ax2.set_xlabel("layer", fontsize=7.5)
    axes[0, 0].set_ylabel("probe AUC", fontsize=7.5)
    axes[1, 0].set_ylabel("steering efficiency", fontsize=7.5)
    for ax in axes.flat:
        ax.tick_params(labelsize=6.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save(fig, "fig_layers.pdf")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(GEN, "layer_summary.csv"), index=False)
    num("layerScans", len(df))
    num("layerDisagree", int((df.lp != df.lsteer).sum()))
    num("layerFracMin", f"{100 * df.frac.min():.0f}")
    num("layerFracMax", f"{100 * df.frac.max():.0f}")
    g = df.iloc[0]
    num("layerGptProbe", g.lp)
    num("layerGptSteer", g.lsteer)
    num("layerGptAUC", f"{g.auc:.3f}")
    num("layerGptLoss", f"{100 * (1 - g.frac):.0f}")
    q = pd.read_csv(os.path.join(RES, "layer_scan_toxicity_Qwen2.5-1.5B-Instruct.csv"))
    num("qtoxLfifteenPPL", f"{q[q.layer == 15].ppl_pct.iloc[0]:.1f}")
    num("qtoxLfifteenEff", f"{q[q.layer == 15].effect.iloc[0]:+.3f}")
    num("qtoxLtenPPL", f"{q[q.layer == 10].ppl_pct.iloc[0]:.1f}")
    num("qtoxLtenEff", f"{q[q.layer == 10].effect.iloc[0]:+.3f}")
    lines = [r"\begin{tabular}{@{}l c cc cc c@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Probe-AUC optimum} & "
             r"\multicolumn{2}{c}{Steering optimum} & \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Scan & Layers & Layer & AUC & Layer & Effect / ppl & "
             r"Eff.\ retained at probe optimum \\", r"\midrule"]
    for _, rr in df.iterrows():
        lines.append(f"{rr.scan} & {rr.n} & {rr.lp} & {rr.auc:.3f} & {rr.lsteer} & "
                     f"${rr.eff_at:+.3f}$\\,/\\,{rr.ppl_at:.1f}\\% & {100 * rr.frac:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_layers.tex"), "w").write("\n".join(lines))


# ===========================================================================
# 8. Degeneration check (distinct-2 of generations)
# ===========================================================================
def distinct2(t) -> float:
    w = str(t).split()
    if len(w) < 3:
        return 1.0
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / len(bg)


def degeneration() -> None:
    lines = [r"\begin{tabular}{@{}l ccc@{}}", r"\toprule",
             r"Run & Unsteered distinct-2 & corr(effect, distinct-2) & "
             r"Diff.-in-means distinct-2 at $\alpha{=}4$ \\", r"\midrule"]
    corrs = []
    for r in RUNS:
        if r["axis"] != "Sentiment":
            continue                  # toxicity generations are not reproduced in the paper
        p = os.path.join(RES, f"generations_{r['tag']}.csv")
        if not os.path.exists(p):
            continue
        g = pd.read_csv(p)
        g["d2"] = g.completion.map(distinct2)
        rep = g.groupby(["arm", "alpha"]).d2.mean().reset_index()
        s = steer(r["tag"]).groupby(["arm", "alpha"]).effect.mean().reset_index()
        m = s.merge(rep, on=["arm", "alpha"])
        c = float(np.corrcoef(m.effect, m.d2)[0, 1])
        corrs.append(c)
        base = float(m[m.alpha == 0].d2.mean())
        dim4 = m[(m.arm == "diff_in_means") & (m.alpha == 4.0)].d2
        lines.append(f"{r['label']} & {base:.3f} & ${c:+.2f}$ & "
                     f"{(dim4.iloc[0] if len(dim4) else float('nan')):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_degeneration.tex"), "w").write("\n".join(lines))
    num("degenCorrMax", f"{max(corrs, key=abs):+.2f}")


# ===========================================================================
# 9. Qualitative examples (sentiment only; LaTeX-escaped)
# ===========================================================================
def tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}
    s = "".join(rep.get(ch, ch) for ch in str(s))
    return "".join(ch for ch in s if ord(ch) < 0x2000 or ch in "–—")


def qualitative() -> None:
    g = pd.read_csv(os.path.join(RES, "generations_sentiment_gpt2_L7.csv"))
    g = g[g.seed == g.seed.min()]
    prompts = list(dict.fromkeys(g.prompt))[:2]
    cfgs = [("diff_in_means", 0.0, "Unsteered"),
            ("cgsae@0.10/block", 1.0, r"CG-SAE block, $\alpha=1$"),
            ("vanilla/free@1024", 1.0, r"SAE-SSV, $n=1024$, $\alpha=1$"),
            ("diff_in_means", 2.0, r"Diff.-in-means, $\alpha=2$"),
            ("diff_in_means", 8.0, r"Diff.-in-means, $\alpha=8$")]
    lines = [r"\begin{tabular}{@{}p{0.26\linewidth}p{0.68\linewidth}@{}}", r"\toprule"]
    for pi, pr in enumerate(prompts):
        lines.append(r"\multicolumn{2}{@{}l}{\textbf{Prompt:} \emph{" + tex_escape(pr) + r"}} \\")
        lines.append(r"\midrule")
        for arm, al, lab in cfgs:
            row = g[(g.arm == arm) & (g.alpha == al) & (g.prompt == pr)]
            if row.empty:
                continue
            txt = tex_escape(str(row.completion.iloc[0])[:170].replace("\n", " "))
            lines.append(f"{lab} & {txt} \\\\")
        if pi < len(prompts) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_qualitative.tex"), "w").write("\n".join(lines))


# ===========================================================================
# 10. Run inventory (hyperparameters actually used, read from run configs)
# ===========================================================================
def inventory() -> None:
    lines = [r"\begin{tabular}{@{}l l c c c c c@{}}", r"\toprule",
             r"Run & Model & Layer & Seeds & Tokens & $d_{\text{latent}}$ & Unsteered PPL \\",
             r"\midrule"]
    tok = []
    for r in RUNS:
        c = config(r["tag"])
        a = c.get("args", {})
        lines.append(f"{r['key']}{'$^{\\dagger}$' if not r['primary'] else ''} & "
                     f"{r['model']} & {r['layer']} & {len(a.get('seeds', []))} & "
                     f"{c.get('n_tokens', 0):,} & {c.get('d_latent', 0):,} & "
                     f"{c.get('baseline', {}).get('ppl', float('nan')):.2f} \\\\")
        tok.append(c.get("n_tokens", 0))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_inventory.tex"), "w").write("\n".join(lines).replace(",", "{,}"))
    num("nRuns", len(RUNS))
    num("nPrimaryRuns", len(PRIMARY))


# ===========================================================================
def main() -> None:
    exp1()
    penalty_forms()
    sae_quality()
    block()
    comparisons()
    tradeoff()
    layers()
    degeneration()
    qualitative()
    inventory()
    num("budget", f"{BUDGET:.0f}")
    with open(os.path.join(GEN, "numbers.tex"), "w") as f:
        f.write("% Auto-generated by make_paper_assets.py -- do not edit by hand.\n")
        for k in sorted(NUM):
            f.write(f"\\newcommand{{\\{k}}}{{{NUM[k]}}}\n")
    print(f"wrote {len(NUM)} numbers, "
          f"{len([f for f in os.listdir(GEN) if f.startswith('tab_')])} tables, "
          f"{len([f for f in os.listdir(FIG) if f.endswith('.pdf')])} figures")


if __name__ == "__main__":
    main()
