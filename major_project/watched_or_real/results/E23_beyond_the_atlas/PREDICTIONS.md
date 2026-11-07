# E23 preregistration: what do attacks move that the belief atlas misses?

Committed before any full E23 run. The notebook (code/e23_make.py, code/e23_build.py) is frozen in the same commit.

## Motivation
E22: atlas belief readings predict which attacks work on Ministral and Qwen, but nothing does on Llama, and the E21 probe showed
Llama's fiction attacks do not act through the "real" belief. Either Llama's attacks move something outside the atlas, or
nothing linear at the decision token carries the effect.

## Method (layer = each model's preselected layer)
- Coverage: share of the squared norm of (a) benign framing shifts (50 Alpaca tasks x 8 wrappers, wrapped minus plain) and
  (b) harmful framing shifts (100 JBB goals x 8 wrappers, wrapped minus bare) lying in the 21-d span of the 20 atlas beliefs + harm
  direction; random 21-d baseline = 21/D.
- Discovery uses benign data only: project the atlas+harm span out of the benign framing shifts, take the top 8 right singular
  vectors (uncentred). No harmful prompt or label is used to find them.
- Selection: on goals 0-49 (discovery half), each component is oriented so complied attacks project higher (natural-run labels),
  and the 3 with the highest compliance AUROC are chosen.
- Reference: a supervised residual compliance direction (complied minus refused attack states on goals 0-49, atlas span removed).
- Causal test on held-out goals 50-99, at the decision position, at the chosen layer only:
  - up: plain harmful requests (n=50) clamped to (complied mean + gap) along the direction;
  - down: fiction attacks (n=200) clamped to (refused mean - gap);
  - each against the same-size move along a random direction orthogonal to the atlas span and the direction.
  - Selective effect = ASR(direction) - ASR(random) for up, ASR(random) - ASR(direction) for down; bootstrap over items.
- Judge: HarmBench Llama-2-13B classifier, as in all earlier experiments.

## Hypotheses
- H1 (coverage gap): the atlas+harm span holds a smaller share of the benign framing shift on Llama than on Ministral.
- H2 (Llama has an unmapped framing direction): on Llama, at least one of the 3 selected framing components has a positive
  selective effect (up or down) with one-sided bootstrap p < 0.05/6 (Bonferroni over 3 components x 2 tests).
- H3 (generality): the H2 criterion holds on at least 2 of the 4 models.
- H4 (refusal intent vs framing): on at least 3 of 4 models the supervised direction's best selective effect exceeds that of the
  best framing component, while its benign framing load is below that of the top framing component.

Every hypothesis is reported as held or failed exactly as stated. Directions are described by wrapper loadings and logit lens,
and that description is exploratory.
