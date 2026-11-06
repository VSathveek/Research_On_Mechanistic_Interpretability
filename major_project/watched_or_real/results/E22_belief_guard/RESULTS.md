# E22 results: belief-manipulation guard (preregistration in PREDICTIONS.md, commit a8b78ec)

## Deviations from the frozen analysis (both forced by crashes, made before reading any affected number)
1. Inner threshold CV uses min(5, positives) folds, and in-sample scores if a training fold has < 2 unsafe prompts.
   Needed on Llama/Gemma, where holding out fiction leaves very few unsafe training examples (Llama: 7 and 1).
   Ministral never hit this case, and its numbers are identical before and after the change.
2. Gemma raw states were saved in fp16 and one massive-activation dimension overflowed (0.007% of entries); clipped to +-65504.
   Affects only the raw-probe baseline on Gemma.

## Main numbers (leave-one-attack-family-out x goal-half-out)
| model (unsafe n) | compliance AUROC: belief / harm / Llama Guard / raw | recall@5% benign: belief / Llama Guard / belief+LG | realised benign block (belief) |
|---|---|---|---|
| Ministral (287) | 0.81 / 0.21 / 0.36 / 0.72 | 0.64 / 0.90 / 0.75 | 15.5% |
| Llama (95) | 0.23 / 0.21 / 0.36 / 0.35 | 0.00 / 0.76 / 0.00 | 0.3% |
| Qwen (71) | 0.65 / 0.43 / 0.22 / 0.64 | 0.10 / 0.78 / 0.09 | 0.3% |
| Gemma (41) | 0.74 / 0.12 / 0.36 / 0.63 | 0.27 / 0.85 / 0.34 | 5.6% |

## Preregistered hypotheses
- H1 belief > harm on compliance AUROC (CI excl. 0), >= 3/4: Ministral +0.60, Qwen +0.22, Gemma +0.62 hold; Llama +0.02 [-0.04, +0.08] fails. HELD (3/4).
- H2 belief > Llama Guard on compliance AUROC, >= 3/4: Ministral +0.45, Qwen +0.42, Gemma +0.38 hold; Llama -0.12 fails. HELD (3/4).
- H3 belief+LG > LG on recall@5%, >= 3/4: negative on all four (-0.15, -0.76, -0.69, -0.51). FAILED (0/4).
- H4 belief recall@5% >= LG, >= 3/4: lower on all four. FAILED (0/4).
- H5 permission recall < fiction recall on Ministral and Qwen: Ministral 0.18 < 0.75 holds; Qwen 0.27 > 0.00 fails. FAILED.
- H6 belief compliance AUROC >= raw - 0.05, >= 3/4: Ministral, Qwen, Gemma hold; Llama -0.12 fails. HELD (3/4).

## Exploratory (NOT preregistered): is H1/H2 just a sign flip?
Zero-shot harm and Llama Guard scores rank REFUSED attacks as more harmful (AUROC < 0.5), and a fitted guard can learn that sign.
Same folds and regression, fitted single- and two-feature baselines (e22_explore.txt), belief minus baseline on compliance AUROC:
- vs fitted harm: Ministral +0.61, Qwen +0.23, Gemma -0.12, Llama -0.05 (n.s.)
- vs fitted harm + Llama Guard: Ministral +0.22, Qwen +0.29, Gemma -0.11, Llama -0.07
So the belief readings carry compliance information beyond content on Ministral and Qwen only. On Gemma one harm reading
(with its sign learned) does better, and on Llama nothing generalises across attack families.

## What this means
- As a deployable guard, the belief guard loses clearly to Llama Guard on every model, and adding beliefs to Llama Guard makes it
  worse when the attack family is new. Its threshold also fails to transfer (Ministral blocks 15.5% of benign instead of 5%).
- As a monitor of WHICH attacks a model will fall for, situational-belief readings beat content-only signals on 2 of 4 models
  under the strictest comparison. Where they work is the same split as before: strongest on Ministral, absent on Llama.
