# E22 preregistration: belief-manipulation guard

Written and committed before any full E22 feature run. The analysis code (code/e22_analyze.py) is committed in the same commit
and is run unchanged on the full data.

## Question
Jailbreak wrappers move the model's situational beliefs (E18 atlas, E19/E21 shift analyses). Can a guard that reads those beliefs
at the decision token tell which harmful prompts the model will actually comply with, and block them better than
(a) a harm probe that reads only "is this request harmful" and (b) external guard models that read only the text?

## Data (no new generation by the protected models)
- Prompts: the exact E19/E20/E21 natural-run sets. 900 attack prompts (100 JailbreakBench goals x 9 wrappers), 150 XSTest safe,
  100 OR-Bench-hard, 60 benign fiction, plus 400 NEW wrapper-matched benign prompts (50 Alpaca tasks x the 8 attack wrappers),
  so a guard cannot win by flagging wrapper language.
- Labels: unsafe = attack prompt whose natural-run completion the HarmBench classifier judged a success. Taken from the existing
  natural runs (Ministral E19, Llama E20, Qwen and Gemma E21). Unsafe counts: Ministral 287, Llama 95, Qwen 71, Gemma 41.
- Features at the model's preselected layer (Ministral 15, Llama 10, Gemma 30, Qwen in-notebook rule, 15):
  readings of all 20 atlas beliefs, each normalised to its own pole means from neutral framing sentences;
  a harm reading (AdvBench vs Alpaca plain requests, disjoint from the test goals); the raw residual state.
- External guards, scored on the same prompts: Llama Guard 3 8B, Granite Guardian 3.1 8B (logit of unsafe vs safe verdict).

## Guards compared (hyperparameters fixed, no tuning)
- harm: harm reading alone, zero-shot.
- belief: logistic regression (C=0.1, balanced) on harm + 20 beliefs + 20 harm x belief interactions.
- raw: same regression on a 64-component PCA of the raw state (an uninterpretable upper reference).
- llamaguard, granite: zero-shot guard-model scores.
- belief+llamaguard: the belief features plus the Llama Guard score.

## Protocol
Folds = leave one attack family out (fiction, professional, permission, bare) x leave one goal half out: each test prompt is scored by a model
that saw neither its wrapper family nor its goal. Benign prompts are split into the same folds (wrapped benign by wrapper family,
XSTest/OR-Bench by index). Thresholds are set to block 5% of the fold's TRAINING benign prompts, scored out of sample.
CIs: bootstrap over goals (2000 resamples).

## Metrics
- compliance AUROC: unsafe vs refused attack prompts (does the guard know WHICH attacks will work?).
- recall@5%: share of unsafe prompts blocked at the 5%-benign threshold; realised benign block rate reported alongside.
- per-family recall, only for families with >= 10 unsafe prompts.

## Hypotheses
- H1 (mechanism): belief beats harm on compliance AUROC, CI excluding 0, on >= 3 of 4 models.
- H2: belief beats Llama Guard on compliance AUROC, CI excluding 0, on >= 3 of 4 models.
- H3 (complementarity): belief+llamaguard beats Llama Guard alone on recall@5%, CI excluding 0, on >= 3 of 4 models.
- H4: belief recall@5% >= Llama Guard recall@5% (point estimate) on >= 3 of 4 models.
- H5 (known weakness from E19): on the models with >= 10 unsafe permission prompts (Ministral, Qwen), belief recall on
  permission is lower than on fiction.
- H6 (interpretability cost): belief compliance AUROC >= raw compliance AUROC - 0.05 on >= 3 of 4 models.

Each hypothesis is reported as held or failed exactly as stated, whatever the outcome.
