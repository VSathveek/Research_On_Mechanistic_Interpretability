"""
Assets for the CARE 2026 extended abstract: the in-model bucket figure and number macros.

Every in-model number quoted in main_ea.tex comes from here (results/bucket_llm_*.csv).
The SAE-side numbers are reused from ../paper/generated/numbers.tex, so the extended
abstract and the full paper can never disagree.

Usage:  python make_ea_assets.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
os.makedirs(os.path.join(HERE, "generated"), exist_ok=True)

C_NEG, C_POS, C_MUTED, C_INK = "#2b2b2b", "#bdbdbd", "#555555", "#000000"

SETTINGS = [  # (macro prefix, label, file, model name)
    ("gl", "GPT-2, λ = 1", "bucket_llm_finetune.csv", "bucket_l1"),
    ("gt", "GPT-2, λ = 3", "bucket_llm_finetune_l3.csv", "bucket_l3"),
    ("gh", "GPT-2, λ = 10", "bucket_llm_finetune.csv", "bucket_l10"),
    ("ss", "From scratch, small", "bucket_llm_scratch.csv", "bucket_l1"),
    ("sl", "From scratch, large", "bucket_llm_scratch_large.csv", "bucket_l1"),
]
PRUNES = [("B", "unwanted"), ("G", "wanted"), ("random", "random")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

NUM = {}


def num(k, v):
    assert k.isalpha(), k
    NUM[k] = v


def stats(f, model):
    d = pd.read_csv(os.path.join(RES, f))
    out = {}
    for pr, _ in PRUNES:
        rows = []
        for _, g in d[d.model == model].groupby("seed"):
            b = g[g.prune == "none"].iloc[0]
            x = g[g.prune == pr].iloc[0]
            rows.append([100 * (x.ppl_imdb_neg / b.ppl_imdb_neg - 1),
                         100 * (x.ppl_imdb_pos / b.ppl_imdb_pos - 1),
                         x.neg_on_negprompts - b.neg_on_negprompts])
        out[pr] = np.array(rows)
    meta = dict(
        seeds=d.seed.nunique(),
        routing=d[d.model == model].routing.mean(),
        wiki_b=d[(d.model == model) & (d.prune == "none")].ppl_wiki.mean(),
        wiki_p=d[(d.model == "plain") & (d.prune == "none")].ppl_wiki.mean(),
        rel_b=d[(d.model == model) & (d.prune == "B")].neg_after_relearn.mean(),
        rel_r=d[(d.model == model) & (d.prune == "random")].neg_after_relearn.mean(),
    )
    return out, meta


def main():
    words = {"B": "Bad", "G": "Good", "random": "Rand"}
    all_stats = {}
    for key, label, f, m in SETTINGS:
        s, meta = stats(f, m)
        all_stats[key] = (label, s, meta)
        num(f"{key}Seeds", meta["seeds"])
        num(f"{key}Routing", f"{meta['routing']:.2f}")
        num(f"{key}WikiBucket", f"{meta['wiki_b']:.0f}")
        num(f"{key}WikiPlain", f"{meta['wiki_p']:.0f}")
        num(f"{key}RelearnBad", f"{meta['rel_b']:.2f}")
        num(f"{key}RelearnRand", f"{meta['rel_r']:.2f}")
        for pr, _ in PRUNES:
            r = s[pr]
            num(f"{key}{words[pr]}Neg", f"{r[:, 0].mean():+.1f}")
            num(f"{key}{words[pr]}Pos", f"{r[:, 1].mean():+.1f}")
            num(f"{key}{words[pr]}NegLo", f"{r[:, 0].min():+.1f}")
            num(f"{key}{words[pr]}NegHi", f"{r[:, 0].max():+.1f}")
            num(f"{key}{words[pr]}Gen", f"{r[:, 2].mean():+.3f}")
    gt = all_stats["gt"][2]
    num("gtWikiCostPct", f"{100 * (gt['wiki_b'] / gt['wiki_p'] - 1):.0f}")
    gh = all_stats["gh"][2]
    num("ghWikiCostPct", f"{100 * (gh['wiki_b'] / gh['wiki_p'] - 1):.0f}")

    # ---- figure: selective knowledge loss, one panel per setting ----
    fig, axes = plt.subplots(1, 5, figsize=(6.6, 1.95))
    x = np.arange(3)
    w = 0.38
    for ax, (key, label, f, m) in zip(axes, SETTINGS):
        _, s, meta = all_stats[key]
        neg = [s[p][:, 0].mean() for p, _ in PRUNES]
        pos = [s[p][:, 1].mean() for p, _ in PRUNES]
        nerr = [[s[p][:, 0].mean() - s[p][:, 0].min() for p, _ in PRUNES],
                [s[p][:, 0].max() - s[p][:, 0].mean() for p, _ in PRUNES]]
        perr = [[s[p][:, 1].mean() - s[p][:, 1].min() for p, _ in PRUNES],
                [s[p][:, 1].max() - s[p][:, 1].mean() for p, _ in PRUNES]]
        ax.bar(x - w / 2 - 0.01, neg, w, yerr=nerr, color=C_NEG, edgecolor="black", linewidth=0.5, label="negative reviews",
               error_kw=dict(lw=0.6, capsize=1.3))
        ax.bar(x + w / 2 + 0.01, pos, w, yerr=perr, color=C_POS, edgecolor="black", linewidth=0.5, hatch="///", label="positive reviews",
               error_kw=dict(lw=0.6, capsize=1.3))
        ax.axhline(0, color=C_MUTED, lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in PRUNES], fontsize=6.3)
        ax.set_title(f"{label}\nrouting {meta['routing']:.2f}", fontsize=7.2)
    axes[0].set_ylabel("perplexity increase (%)")
    fig.supxlabel("bucket deleted", fontsize=7.5, y=0.07)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(w_pad=0.5)
    fig.savefig(os.path.join(HERE, "figures", "fig_inmodel.pdf"))
    fig.savefig(os.path.join(HERE, "figures", "fig_inmodel.png"), dpi=200)
    plt.close(fig)

    with open(os.path.join(HERE, "generated", "ea_numbers.tex"), "w") as fh:
        fh.write("% Auto-generated by make_ea_assets.py -- do not edit by hand.\n")
        for k in sorted(NUM):
            fh.write(f"\\newcommand{{\\{k}}}{{{NUM[k]}}}\n")
    print(f"wrote {len(NUM)} macros and fig_inmodel.pdf")


def fig_toy():
    """Greyscale toy-network figure for the extended abstract (wanted / unwanted wording)."""
    raw = pd.read_csv(os.path.join(RES, "exp1_synthetic_raw.csv"))
    summ = pd.read_csv(os.path.join(RES, "exp1_synthetic_summary.csv"))

    def br(t):
        n = max(1, int(32 * t))
        return (32 - 2 * n) / (2 * n)

    rows = []
    for (lam, tf), g in raw.groupby(["lambda", "tag_frac"]):
        b = g[g.ablation == "B-block"].d_auc.mean()
        r = g[g.ablation == "random"].d_auc.mean()
        rows.append((lam, tf, b / r))
    sp = pd.DataFrame(rows, columns=["lam", "tau", "spec"])

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    style = {0.0: ("#9a9a9a", "x", "--", "no penalty (λ = 0)"),
             1.0: ("#555555", "o", "-", "λ = 1"),
             5.0: ("#000000", "s", "-", "λ = 5")}
    for lam, (c, mk, ls, lab) in style.items():
        g = sp[sp.lam == lam].sort_values("tau")
        ax.plot(g.tau, g.spec, color=c, marker=mk, ls=ls, lw=1.4, ms=4.5, label=lab)
    ax.axhline(1.0, color="#777777", ls=":", lw=0.9)
    ax.text(0.10, 0.93, "no better than random (S = 1)", ha="left", va="top", fontsize=6.8,
            color="#555555")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 25)
    ax.set_xlabel("tag fraction τ   (τ = 0.5 means no buffer)")
    ax.set_ylabel("specificity S (log scale)")
    ticks = sorted(sp.tau.unique())
    ax.set_xticks(ticks)
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(ticks)
    sec.set_xticklabels([f"{br(t):.2f}" for t in ticks], fontsize=6.8)
    sec.set_xlabel("buffer size (neutral units per tagged unit)", fontsize=7, color="#555555")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    ax.set_title("(a) Specificity needs the buffer", loc="left", fontsize=8)

    ax = axes[1]
    d = summ[(summ["lambda"] == 1.0) & np.isclose(summ.tag_frac, 0.20)].set_index("ablation")
    order = [("B-block", "unwanted\nbucket"), ("G-block", "wanted\nbucket"),
             ("random", "random\nbucket")]
    x = np.arange(3)
    w = 0.36
    unw = [d.loc[k, "d_unwanted"] for k, _ in order]
    wan = [d.loc[k, "d_wanted"] for k, _ in order]
    ax.bar(x - w / 2 - 0.01, unw, w, color="#2b2b2b", edgecolor="black", lw=0.5,
           label="unwanted class")
    ax.bar(x + w / 2 + 0.01, wan, w, color="#bdbdbd", edgecolor="black", lw=0.5, hatch="///",
           label="wanted class")
    for xi, (u, v) in enumerate(zip(unw, wan)):
        for xx, val in [(xi - w / 2, u), (xi + w / 2, v)]:
            ax.text(xx, val + (0.015 if val >= 0 else -0.015), f"{val:+.2f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=6.5)
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in order], fontsize=6.8)
    ax.set_ylabel("change in accuracy after deletion")
    ax.set_ylim(-0.6, 0.55)
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    ax.set_title("(b) Each bucket removes only its own class", loc="left", fontsize=8)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(HERE, "figures", "fig_toy.pdf"))
    fig.savefig(os.path.join(HERE, "figures", "fig_toy.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
    fig_toy()


def fig_inmodel_compact():
    """Three-panel version of the in-model figure for the 1-page extended abstract."""
    pick = [("gl", "GPT-2, λ = 1 (no buckets form)"), ("gt", "GPT-2, λ = 3"),
            ("sl", "GPT-style model trained from scratch")]
    files = {k: (f, m) for k, _, f, m in SETTINGS}
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.45))
    x = np.arange(3)
    w = 0.38
    for ax, (key, title) in zip(axes, pick):
        s, _ = stats(*files[key])
        neg = [s[p][:, 0].mean() for p, _ in PRUNES]
        pos = [s[p][:, 1].mean() for p, _ in PRUNES]
        ax.bar(x - w / 2 - 0.01, neg, w, color=C_NEG, edgecolor="black", linewidth=0.5,
               label="negative reviews")
        ax.bar(x + w / 2 + 0.01, pos, w, color=C_POS, edgecolor="black", linewidth=0.5,
               hatch="///", label="positive reviews")
        ax.axhline(0, color=C_MUTED, lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"delete {lab}" for _, lab in PRUNES], fontsize=6.3)
        ax.set_title(title, fontsize=7.2)
        ax.tick_params(axis="y", labelsize=6.3)
    axes[0].set_ylabel("perplexity rise (%)", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", frameon=False, fontsize=6.8, bbox_to_anchor=(0.995, 0.55))
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(HERE, "figures", "fig_inmodel_compact.pdf"))
    fig.savefig(os.path.join(HERE, "figures", "fig_inmodel_compact.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_inmodel_compact()
