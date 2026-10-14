# Generates every figure in the paper from saved attributions and the numbers reported in the run logs (results/*/log.txt).
# Usage: python make_figures.py [path_to_attribution_dir]   (default: ../results)
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
# palette (validated reference instance): slot 1 blue = "watched", slot 2 orange = "real"; controls in neutral gray
BLUE, ORANGE, GRAY, LIGHT = "#2a78d6", "#eb6834", "#a3a29c", "#d9d8d3"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e7e6e2", "#ffffff"
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#f2f1ee", "#e34948"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=200); plt.close(fig)
def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h
def err(ps, n):
    lo, hi = zip(*[wilson(p, n) for p in ps]); ps = np.array(ps); return np.vstack([ps - np.array(lo), np.array(hi) - ps])

# ---------- Fig 1: method overview ----------
def box(ax, x, y, w, h, title, sub=None, fc="#f7f7f5", ec=LIGHT, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2 + (0.022 if sub else 0), title, ha="center", va="center", fontsize=7.6, color=tc, weight="bold")
    if sub: ax.text(x + w / 2, y + h / 2 - 0.032, sub, ha="center", va="center", fontsize=7, color=INK2)
def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=INK2, lw=1))
fig, ax = plt.subplots(figsize=(7.6, 3.3)); ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
box(ax, 0.00, 0.72, 0.21, 0.16, '"This is a real request"', "framing R+", "#fdece5", ORANGE, INK)
box(ax, 0.00, 0.52, 0.21, 0.16, '"This is a simulation"', "framing R−", "#fdece5", ORANGE, INK)
box(ax, 0.00, 0.32, 0.21, 0.16, '"You are being watched"', "framing W±", "#e7f0fb", BLUE, INK)
box(ax, 0.00, 0.12, 0.21, 0.16, "+ the same task", "Alpaca / AdvBench / OR-Bench")
box(ax, 0.29, 0.30, 0.22, 0.42, "")
ax.text(0.40, 0.62, "Language model", ha="center", va="center", fontsize=7.6, weight="bold", color=INK)
ax.text(0.40, 0.48, "residual stream h\n~10 late attention heads\nwrite the framing signal", ha="center", va="center", fontsize=6.8, color=INK2)
for y in [0.80, 0.60, 0.40, 0.20]: arrow(ax, (0.215, y), (0.285, 0.51))
box(ax, 0.565, 0.60, 0.22, 0.16, '"real" direction', "diff-in-means, last token", "#fdece5", ORANGE)
box(ax, 0.565, 0.26, 0.22, 0.16, '"watched" direction', "diff-in-means, last token", "#e7f0fb", BLUE)
arrow(ax, (0.515, 0.58), (0.57, 0.68)); arrow(ax, (0.515, 0.44), (0.57, 0.34))
box(ax, 0.83, 0.40, 0.19, 0.20, "Behaviour", "comply / answer / refuse")
arrow(ax, (0.78, 0.68), (0.825, 0.53)); arrow(ax, (0.78, 0.34), (0.825, 0.47))
ax.text(0.5, 0.03, "Tests:  decode (AUROC) · steer  h + αg·d̂ · ablate  h − (h·d̂)d̂ · attribute heads · knock out / patch heads · 2×2 factorial",
        ha="center", fontsize=7.5, color=INK)
save(fig, "fig1_method")

# ---------- Fig 2: decodability and distinctness ----------
models = ["Ministral-8B", "Llama-3.1-8B", "Gemma-3-4B", "Qwen2.5-14B"]
aW, aR = [0.995, 0.88, 0.78, 0.980], [0.993, 0.785, 0.94, 0.971]
cosWR, cosRref = [0.34, 0.21, 0.22, 0.12], [0.10, 0.04, 0.17, -0.02]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.4, 1]})
x = np.arange(4); w = 0.36
axs[0].bar(x - w / 2 - 0.01, aW, w, color=BLUE, label="watched direction", zorder=3)
axs[0].bar(x + w / 2 + 0.01, aR, w, color=ORANGE, label="real direction", zorder=3)
for i in range(4):
    axs[0].text(x[i] - w / 2 - 0.01, aW[i] + 0.008, f"{aW[i]:.2f}", ha="center", fontsize=7, color=INK2)
    axs[0].text(x[i] + w / 2 + 0.01, aR[i] + 0.008, f"{aR[i]:.2f}", ha="center", fontsize=7, color=INK2)
axs[0].axhline(0.5, color=INK2, lw=0.8, ls=":"); axs[0].text(3.55, 0.51, "chance", fontsize=7, color=INK2, ha="right")
axs[0].set_ylim(0.45, 1.16); axs[0].set_xticks(x, models, fontsize=7.5); axs[0].set_ylabel("held-out AUROC")
axs[0].set_title("(a) Decodable on unseen tasks and wordings", fontsize=9, loc="left"); axs[0].legend(fontsize=7, loc="upper center", ncol=2)
axs[1].bar(x - w / 2 - 0.01, cosWR, w, color=GRAY, label="cos(watched, real)", zorder=3)
axs[1].bar(x + w / 2 + 0.01, cosRref, w, color=ORANGE, alpha=0.55, label="cos(real, refusal)", zorder=3)
axs[1].axhline(0, color=INK2, lw=0.8); axs[1].set_ylim(-0.1, 0.45)
axs[1].set_xticks(x, ["Min.", "Llama", "Gemma", "Qwen"], fontsize=7.5); axs[1].set_ylabel("cosine similarity")
axs[1].set_title("(b) Distinct from each other and refusal", fontsize=9, loc="left"); axs[1].legend(fontsize=7, loc="upper right")
fig.tight_layout(); save(fig, "fig2_geometry")

# ---------- Fig 3: factorial selectivity ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=False)
for ax, (m, rR, rW, wR, wW, sr, sw) in zip(axs, [("Llama-3.1-8B", 0.321, 0.041, 0.075, 0.214, 7.8, 2.9), ("Ministral-8B", 0.751, 0.167, 0.258, 0.532, 4.5, 2.1)]):
    xs = np.arange(2); w = 0.36
    ax.bar(xs - w / 2 - 0.01, [rR, wR], w, color=ORANGE, label="effect of real/simulated factor", zorder=3)
    ax.bar(xs + w / 2 + 0.01, [rW, wW], w, color=BLUE, label="effect of watched/unwatched factor", zorder=3)
    ax.set_xticks(xs, ['projection on\n"real" direction', 'projection on\n"watched" direction'], fontsize=7.5)
    ax.set_title(f"{m}: {sr}× and {sw}× selective", fontsize=9, loc="left"); ax.set_ylabel("main effect on projection")
axs[0].legend(fontsize=7, loc="upper right")
fig.tight_layout(); save(fig, "fig3_factorial")

# ---------- Fig 4: steering on natural prompts (E10) ----------
labels = ["toward\n\"real\" ×4", "toward\n\"real\" ×2", "no\nsteering", "random\nvector a", "random\nvector b", "toward\n\"simulated\" ×2", "toward\n\"simulated\" ×4"]
vals = [0.002, 0.015, 0.042, 0.063, 0.056, 0.156, 0.287]
cols = [ORANGE, ORANGE, GRAY, LIGHT, LIGHT, ORANGE, ORANGE]
fig, ax = plt.subplots(figsize=(7.6, 2.9))
ax.bar(range(7), np.array(vals) * 100, color=cols, width=0.62, zorder=3, yerr=err(vals, 520) * 100, error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 2})
for i, v in enumerate(vals): ax.text(i, v * 100 + (err(vals, 520)[1][i] * 100) + 0.8, f"{v*100:.1f}%", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(range(7), labels, fontsize=7); ax.set_ylabel("harmful compliance (%)"); ax.set_ylim(0, 38)
ax.set_title("Steering the \"real\" direction on benchmark-formatted prompts, no framing sentence (Ministral, AdvBench n = 520)", fontsize=8.5, loc="left")
save(fig, "fig4_natural_steering")

# ---------- Fig 5: steering dose-response, small multiples ----------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
panels = [("Ministral, AdvBench (E10)", [-4, -2, 0, 2, 4], [0.287, 0.156, 0.042, 0.015, 0.002], 0.063, 520, "harmful compliance (%)"),
          ("Llama, OR-Bench (Granite)", [-4, -2, 0, 2, 4], [0.910, 0.835, 0.675, 0.580, 0.565], 0.855, 200, "answer rate (%)"),
          ("Gemma, OR-Bench (Granite)", [-4, -2, 0, 2, 4], [0.640, 0.655, 0.625, 0.610, 0.510], None, 200, "answer rate (%)")]
for ax, (t, a, v, rnd, n, yl) in zip(axs, panels):
    v = np.array(v); e = err(v, n) * 100
    ax.errorbar(a, v * 100, yerr=e, color=ORANGE, lw=2, marker="o", ms=4.5, capsize=2, zorder=3)
    if rnd is not None: ax.scatter([-4], [rnd * 100], color=GRAY, marker="s", s=28, zorder=4, label="random vector ×4")
    ax.set_xticks(a, ["−4", "−2", "0", "+2", "+4"])
    ax.set_title(t, fontsize=8.5, loc="left"); ax.set_ylabel(yl, fontsize=7.5)
    if rnd is not None: ax.legend(fontsize=6.5, loc="upper right")
fig.supxlabel("steering strength α   (− = toward simulated, + = toward real)", fontsize=8, color=INK); fig.tight_layout(); save(fig, "fig5_dose_response")

# ---------- Fig 6: head attribution heatmaps ----------
def load(p):
    try: return np.load(p)
    except Exception: return None
srcs = [("Ministral-8B (layers 0–14)", os.path.join(RES, "E7_head_localisation_ministral", "head_attr.npy")),
        ("Llama-3.1-8B (layers 0–9)", os.path.join(RES, "E7L_head_localisation_llama31_8b", "head_attr.npy")),
        ("Gemma-3-4B (layers 0–29)", os.path.join(RES, "E7G_head_localisation_gemma3_4b", "head_attr.npy"))]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [32, 32, 14]})
for ax, (t, p) in zip(axs, srcs):
    A = load(p); ax.grid(False)
    if A is None: ax.text(0.5, 0.5, "missing " + os.path.basename(os.path.dirname(p)), ha="center"); continue
    A = A / np.abs(A).sum() * 100; m = np.abs(A).max()
    im = ax.imshow(A, aspect="auto", cmap=DIV, norm=TwoSlopeNorm(0, -m, m), origin="lower", interpolation="nearest")
    top = np.dstack(np.unravel_index(np.argsort(-np.abs(A).ravel())[:3], A.shape))[0]
    for (l, h) in top: ax.text(h, l, f"L{l}.H{h}", fontsize=6, ha="center", va="center", color=INK, weight="bold")
    ax.set_title(t, fontsize=8, loc="left"); ax.set_xlabel("head", fontsize=7.5); ax.set_ylabel("layer", fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label("% of |attention attribution|", fontsize=6)
fig.tight_layout(); save(fig, "fig6_head_attribution")

# ---------- Fig 7: knock-outs vs controls ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
for ax, (t, none, top, ctrl, lay, unit) in zip(axs, [
        ("Ministral, AdvBench n = 520", 3.5, 0.8, [2.3, 3.1, 3.3, 2.7, 3.1, 2.7], [2.1, 4.8], "pts"),
        ("Llama, OR-Bench n = 500", 11.4, 2.2, [10.2, 1.6, 10.0, 11.4, 8.0, 6.4], [10.2], "pts")]):
    ax.axhline(none, color=INK2, lw=1, ls="--"); ax.text(2.45, none + 0.25, f"no knock-out: +{none}", fontsize=7, color=INK2, ha="right")
    ax.scatter(np.full(len(ctrl), 1) + np.linspace(-0.12, 0.12, len(ctrl)), ctrl, color=GRAY, s=26, zorder=3, label="10 random heads")
    ax.scatter(np.full(len(lay), 2) + np.linspace(-0.05, 0.05, len(lay)), lay, color=LIGHT, edgecolor=INK2, s=26, zorder=3, label="10 same-layer heads")
    ax.scatter([0], [top], color=ORANGE, s=60, zorder=4, label="top-10 attributed heads")
    ax.text(0.12, top, f"+{top}", fontsize=7.5, va="center", color=INK)
    ax.set_xticks([0, 1, 2], ["top-10", "random sets", "same-layer\nsets"], fontsize=7.5); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("framing gap (sim − real), pts"); ax.set_ylim(0, max([none] + ctrl + lay) * 1.18); ax.set_title(t, fontsize=8.5, loc="left")
axs[0].legend(fontsize=6.5, loc="lower right")
fig.tight_layout(); save(fig, "fig7_knockouts")

# ---------- Fig 8: cross-prompt patching (Llama, Granite) ----------
fig, ax = plt.subplots(figsize=(4.2, 2.6))
lab = ['"real"\nframing', '"simulated"\nframing', '"simulated" +\ntop-10 heads\npatched to "real"', '"simulated" +\n10 random heads\npatched']
v = [0.464, 0.674, 0.446, 0.674]; c = [ORANGE, GRAY, ORANGE, LIGHT]
ax.bar(range(4), np.array(v) * 100, color=c, width=0.6, zorder=3, yerr=err(v, 500) * 100, error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 2})
for i, x_ in enumerate(v): ax.text(i, x_ * 100 + 5, f"{x_*100:.1f}%", ha="center", fontsize=7.5)
ax.set_xticks(range(4), lab, fontsize=6.8); ax.set_ylabel("answer rate (%)"); ax.set_ylim(0, 85)
ax.set_title("Patching heads with their \"real\" state from other prompts (Llama)", fontsize=8.5, loc="left")
save(fig, "fig8_patching")

# ---------- Fig 9: crossed internal dissociation + shared behavioural pathway (Llama) ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
xs = np.arange(2); w = 0.36
axs[0].bar(xs - w / 2 - 0.01, [56, 13], w, color=ORANGE, label="real/simulated signal", zorder=3)
axs[0].bar(xs + w / 2 + 0.01, [23, 60], w, color=BLUE, label="watched/unwatched signal", zorder=3)
axs[0].set_xticks(xs, ["mean-ablate\ntop-10 heads", 'ablate "watched"\ndirection'], fontsize=7.5); axs[0].set_ylabel("internal signal removed (%)")
axs[0].set_title("(a) Internal: crossed dissociation", fontsize=8.5, loc="left"); axs[0].legend(fontsize=6.5); axs[0].set_ylim(0, 75)
xs = np.arange(3)
axs[1].bar(xs - w / 2 - 0.01, [4.0, 5.7, -2.0], w, color=ORANGE, label="real/simulated effect", zorder=3)
axs[1].bar(xs + w / 2 + 0.01, [11.8, 9.3, 1.2], w, color=BLUE, label="watched/unwatched effect", zorder=3)
axs[1].axhline(0, color=INK2, lw=0.8)
axs[1].set_xticks(xs, ["none", "mean-ablate\ntop-10 heads", 'ablate "watched"\ndirection'], fontsize=7.5); axs[1].set_ylabel("behavioural effect (pts)")
axs[1].set_title("(b) Behaviour: shared pathway", fontsize=8.5, loc="left"); axs[1].legend(fontsize=6.5)
fig.tight_layout(); save(fig, "fig9_dissociation")

# ---------- Fig 10: judge agreement ----------
fig, ax = plt.subplots(figsize=(4.4, 2.3))
rows = [("Ministral, AdvBench\n(HarmBench clf.)", 0.55, 0.65), ("Llama, OR-Bench\n(Granite)", 0.61, 0.72), ("Qwen-14B, OR-Bench\n(Granite)", 0.43, 0.43),
        ("Ministral, OR-Bench\n(Granite), withdrawn", 0.36, 0.36), ("Gemma, OR-Bench\n(Granite), judge replaced", 0.28, 0.40)]
for i, (l, lo, hi) in enumerate(rows):
    ax.plot([lo, hi], [i, i], color=GRAY, lw=3, solid_capstyle="round", zorder=2); ax.scatter([lo, hi], [i, i], color=INK2, s=14, zorder=3)
ax.axvline(0.6, color=INK2, lw=0.8, ls=":"); ax.text(0.61, -0.45, "κ ≥ 0.6: substantial", fontsize=6.5, color=INK2)
ax.set_yticks(range(5), [r[0] for r in rows], fontsize=6.8); ax.set_xlim(0, 0.8); ax.set_xlabel("Cohen's κ with primary judge (range over runs)")
ax.invert_yaxis(); ax.set_title("Agreement between judges", fontsize=8.5, loc="left")
save(fig, "fig10_judges")
print("figures written to", OUT)
