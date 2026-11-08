# E24 preregistration: protect the belief each model actually uses

Committed before any full E24 run, together with the frozen notebook builder (code/e24_make.py, code/e24_build.py) and
analysis (code/e24_analyze.py).

## Question
BSCT that protects the "real" belief beat paper-faithful ACT on Ministral but lost badly on Llama (fiction ASR 17.0 vs 2.2).
E23 found that Llama's jailbreaks run mainly through a role-play framing direction outside the atlas. If BSCT protects THAT
direction instead, does it close the gap on Llama?

## Runs (seed 0, same data, LoRA, schedule and evaluation as E19-E21; one new loss term)
- bsct_dir: consistency loss only along the E23 direction, at the layer it was found, weighted like the 5 window layers of BSCT,
  scaled by that direction's complied-minus-refused gap.
- bsct_combo: the original "real" BSCT loss plus that term.
- Directions (from results/E23_beyond_the_atlas): Llama pc2 (role-play), Ministral pc0 (generic framing), and on each model the
  supervised residual compliance direction (bsct_dir only).
- Six runs: llama dir_pc2, combo_pc2, dir_supervised; min dir_pc0, combo_pc0, dir_supervised.
- Comparison runs already exist with the same seed and prompts: natural, bsct, act_paper (Llama E20, Ministral E19).

## Primary evaluation
Attack prompts on JBB goals 50-99 only (E23 used goals 0-49 to pick and orient directions). Paired McNemar tests and
goal-clustered bootstrap CIs. Benign sets (XSTest, OR-Bench, creative) and MMLU as before.

## Hypotheses
- H1: on Llama, dir_pc2 has lower fiction ASR than BSCT ("real"), one-sided McNemar p < 0.05.
- H2: on Llama, dir_pc2 is non-inferior to paper ACT on fiction ASR (upper 95% CI of dir_pc2 - ACT < +5 points) AND its
  OR-Bench answer rate is at least ACT's (point estimate).
- H3: on Ministral, dir_pc0 is non-inferior to BSCT ("real") on fiction ASR (upper 95% CI of the difference < +5 points).
- H4: on both models, combo fiction ASR <= min(BSCT, dir) (point estimates).
- H5: protecting the supervised compliance direction over-refuses more than protecting the framing direction: lower OR-Bench
  answer rate on both models (point estimates).

Each hypothesis is reported as held or failed exactly as stated.
