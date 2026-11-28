# CG-SAE v2 — rebuild for CARE 2026

Rebuild of the minor project (*Class-Grouped Sparse Autoencoders*, NIT-AP 2025–26) so the
central claim is testable against a real baseline. Target: CARE Conference 2026, IIT Roorkee,
extended abstract due **30 September 2026**.

---

## The question, restated

The v1 report claimed CG-SAE "enables surgical inference-time steering". It had no control
arm, so the claim could not be evaluated. After a literature check
(see [notes/lit_review.md](notes/lit_review.md)) the defensible question is narrower and
sharper:

> Supervised SAEs already exist. Scar (arXiv 2411.07122) conditions **one** latent on a
> concept during training; SAE-SSV (arXiv 2505.16188) selects latents **post-hoc** from a
> pretrained SAE. Neither asks how much latent **capacity** a concept should get, or where
> features shared between the two opposing classes should live.
>
> **Does allocating a block of latents per class, with a neutral buffer between them, beat
> one conditioned latent and post-hoc selection — at matched sparsity and matched
> reconstruction?**

Both answers are publishable. A win is a method result. A loss is an honest negative result
on an assumption three prior papers make without testing.

---

## What was wrong with v1 (all verified in the source)

| # | Problem | Where | Consequence |
|---|---|---|---|
| 1 | `bad_neurons = np.argsort(bb)[-150:]` ranks over **all 768 latents**, not the B-group | notebook cell 13 | The steering and perplexity results never used the class grouping. They *are* the post-hoc baseline. |
| 2 | Unpatched completions sampled (`temperature=0.7, top_p=0.9`); patched completions `argmax` | cells 19, 18 | The 4/10 vs 0/10 shift counts compare two different decoders. |
| 3 | SAE trained on **mean-pooled** sequence vectors, applied to **per-token** residuals | cells 7, 18 | Train/test distribution mismatch; `act_mean`/`act_std` come from the wrong distribution. |
| 4 | `d_latent = d_in = 768`, L0 ≈ 394/768 | cell 10 | Not a sparse autoencoder. Overcompleteness is the mechanism SAEs rely on. |
| 5 | No vanilla-SAE control anywhere | — | The central claim has no baseline. `+2,300% mean-pooling` is a strawman. |
| 6 | Pruning asymmetry measured with per-class accuracy at a fixed threshold | cell 16 | `+0.06 pos / −0.10 neg` is the signature of a shifted decision boundary. Balanced accuracy moves 0.67 → 0.65. |
| 7 | n=100 test set, single seed, `best_tf` selected by the metric then reported as the result | cells 4, 6, 16 | ±10 points is ≈1.4 SE. Nothing is powered. |
| 8 | Perplexity measured on the 10 positive prompts, last-5 positions only | cell 21 | Measured where the damage is smallest; earlier tokens are bit-identical by causality. |
| 9 | Leakage = mean raw off-group activation | cell 11 | Gameable: shrinking all activations 100× divides it by 100. Verified — see below. |

**On (9)**, measured directly:

```
             raw      normalised
scale 1.0  : 0.12498   0.20241
scale 0.1  : 0.01250   0.20241
scale 0.01 : 0.00125   0.20241
```

A pure rescaling that changes nothing about separation drops the v1 metric 100×. This is
why v1's Exp-1 τ=0.20 config scored "best leakage" while its B-group encoded nothing
(Δbad = −0.0000), and why Exp-2 then selected τ=0.20 as the headline configuration.

---

## What v2 changes

- **Per-token activations** (~4×10⁵ tokens) instead of 1,000 mean-pooled vectors.
- **TopK SAEs** (Gao et al., arXiv 2406.04093) at `expansion=8`, so **L0 is exact and
  identical across every arm** — the only way the head-to-head is fair.
- **Delta-form intervention**: `h' = h + (dec(z') − dec(z))·σ`. The SAE reconstruction error
  never enters the residual stream, so α=0 is exactly a no-op and the measured perplexity
  cost is the cost of the *intervention*, not of the autoencoder.
- **Scale-invariant leakage** (off-group share of tagged activation mass).
- **External judges** (distilbert-SST-2, roberta-toxicity) instead of a 35-word vocabulary list.
- **Greedy decoding everywhere**, so arms are comparable and runs deterministic.
- **Off-axis perplexity** (WikiText-103), edited at every position.
- **Threshold-free metrics** (AUC), 3 seeds, 95% CIs.

### Headline evaluation
Every method is swept over intervention strength and scored by **effect at a matched
perplexity budget**. A method that wins only by intervening less cannot win on that plot —
which is precisely what v1's single-operating-point `+31.6%` could not rule out.

---

## Arms

| Arm | Role |
|---|---|
| `vanilla/free` | TopK SAE + post-hoc top-k selection — **the SAE-SSV control** |
| `scar/free` | One label-conditioned latent (BCE) — **the Scar control** |
| `cgsae@τ/group` | CG-SAE, latents selected **inside the B-block** — the only arm that uses the grouping |
| `cgsae@τ/free` | CG-SAE, latents selected post-hoc over all latents — **isolates grouping from selection** (this is what v1 actually did) |
| `*/random` | Random latents of the same count — null control |
| `diff_in_means` | mean(unwanted) − mean(wanted) steering (Turner et al.; Arditi et al.) |
| `probe` | Logistic-regression direction |
| `random_dir` | Random direction — null control |

`diff_in_means` is not a formality: arXiv 2510.01246 finds mean-activation-difference
*beats* SAE steering on several task families.

---

## Layout

```
cgsae/
├── src/
│   ├── cgsae_core.py       SAE variants (TopK / CG / Scar), training, metrics
│   ├── cgsae_model.py      activation capture, interventions, judges, perplexity
│   ├── cgsae_data.py       the three concept axes + off-axis perplexity corpus
│   ├── run_experiment.py   the driver
│   └── analyse.py          figures + the matched-budget table
├── build_and_push.py       assembles src/ into one Kaggle script kernel and pushes it
├── kernels/                generated kernels (one directory per run)
├── notes/lit_review.md     positioning against prior supervised-SAE work
├── results/                CSVs pulled back from Kaggle
└── figures/
```

## Kaggle kernel naming

```
cgsae-v2-<axis>-<model>-l<layer>[-<suffix>]
```

Rebuild and push with:

```bash
python build_and_push.py --axis sentiment --model gpt2 --layer 8
python build_and_push.py --axis toxicity  --model gpt2 --layer 8
python build_and_push.py --axis sentiment --model Qwen/Qwen2.5-1.5B-Instruct --layer 14
```

`--smoke` shrinks everything for a pipeline check; `--no-push` builds locally only.

---

## Run log

All runs on Kaggle. Source of truth for every number is `results/`.

| Kernel | Axis | Model | Layer | Seeds | Status |
|---|---|---|---|---|---|
| `cgsae-v2-exp1-synthetic` | synthetic | 64-D MLP | — | 20 | done (CPU) |
| `cgsae-v2-sentiment-gpt2-l8-penform` | sentiment | gpt2 | 8 | 2 | done — penalty-form table |
| `cgsae-v2-sentiment-gpt2-l8-layerscan` | sentiment | gpt2 | all 12 | 1 | done |
| `cgsae-v2-sentiment-qwen1p5b-l18-layerscan` | sentiment | Qwen2.5-1.5B | all 28 | 1 | done |
| `cgsae-v2-toxicity-gpt2-l8-layerscan` | toxicity | gpt2 | all 12 | 1 | done |
| `cgsae-v2-sentiment-gpt2-l7-main` | sentiment | gpt2 | 7 | 5 | done |
| `cgsae-v2-toxicity-gpt2-l8-main` | toxicity | gpt2 | 8 | 5 | done |
| `cgsae-v2-sentiment-qwen1p5b-l15-main` | sentiment | Qwen2.5-1.5B | 15 | 3 | done |
| `cgsae-v2-sentiment-qwen1p5b-l18-main` | sentiment | Qwen2.5-1.5B | 18 | 3 | done — layer robustness |

Superseded: `-main` at gpt2 L8 (3 seeds, intervention width unmatched) — its apparent
"CG-SAE beats vanilla 2x" did not survive 5 seeds and width matching. Treat as retracted.

---

## Findings

Paired t-tests over seeds, effect measured at a matched 10% perplexity budget.

### 1. The tagged block localises the concept — replicated
Ablating the B-block beats ablating a **size-matched random block** in **10 of 16 tests**,
significant in all four main runs and on both concept axes. tau=0.10 survives Bonferroni
correction in three independent runs. Without the neutral buffer (tau=0.50) the effect is
non-significant in 3 of 4 runs.

### 2. CG-SAE's steering advantage is scale-dependent
Against the vanilla-SAE + post-hoc control (the SAE-SSV recipe), at matched intervention
width: **null on GPT-2** (p=0.24–0.40) but **significant on Qwen2.5-1.5B** — L15 n=1024
+0.277 vs +0.127 (p=0.001), L18 n=256 +0.193 vs +0.098 (p=0.017). Two of four width/layer
tests significant; one survives Bonferroni. Suggestive, not uniform.

### 3. SAE vs difference-in-means is task-dependent
- sentiment: difference-in-means wins at both scales (GPT-2 p=0.001, Qwen p=0.004/0.012)
- **toxicity: the SAE methods win** — Scar +0.194 vs diff-in-means +0.130 (p=0.022)

This reproduces the task-dependence reported in arXiv 2510.01246 rather than contradicting it.

### 4. Group-suppression objectives degenerate under sparsity
v1's squared-mean penalty is inert at real sparsity (leakage 0.494 -> 0.471 across
lambda=1..10, FVU unchanged). Stronger suppression forms "succeed" by silencing the block:
the best-leakage configuration in the whole sweep (`mean`, lambda=10, leakage 0.0017) has a
**96% dead B-block and zero routing**. corr(leakage, routing) = **+0.03** across 12
configurations — the natural metric is uninformative about the property it certifies.
Fixed with a contrastive routing objective plus `routing` / `b_alive_frac` diagnostics.

### 5. The neutral buffer governs specificity (synthetic, 20 seeds)
Specificity = |dAUC B-block| / |dAUC random block of equal width|: **16.6x** at buffer
ratio 4.33, falling monotonically to **~1.0x** with no buffer. At lambda=0 it is ~1.0
everywhere — the partition does nothing until the penalty enforces it. Clean double
dissociation: B-block ablation costs the unwanted class 0.447 accuracy and *gains* 0.184 on
the wanted class; G-block ablation mirrors it; a size-matched random block does neither.

### 6. Choosing the intervention layer by probe accuracy is suboptimal
Replicated in all three layer scans - decodability and interventional efficacy peak at
different depths every time:

| scan | probe AUC peaks | usable steering peaks |
|---|---|---|
| sentiment, GPT-2 (12 layers) | L11 (last), 0.816 | L7 |
| sentiment, Qwen (28 layers) | L16, 0.897 | L15 |
| toxicity, Qwen (28 layers) | L18, 0.786 | L9-L11 |

On GPT-2 selecting by probe AUC costs **45%** of achievable steering efficiency. Qwen's
last layer can match mid-network effect size but needs ~5x the perplexity to do it. The
Qwen toxicity scan is the sharpest case: L15 - the sentiment optimum, and a plausible
carry-over choice - needs **+37.5% perplexity** for an effect L10 reaches at +6.5%.
