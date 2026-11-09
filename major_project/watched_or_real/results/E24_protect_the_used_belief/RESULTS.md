# E24 results: protect the belief each model actually uses

Preregistration: PREDICTIONS.md (commit 5f7f47b). Analysis: code/e24_analyze.py, run unchanged. Full output: e24_stats.txt and
e24_results.json. There are no deviations from the preregistration.

## Primary table (JBB goals 50-99 for attacks; % ASR, or % answered for the benign sets)

Llama-3.1-8B

| run | fiction | permission | professional | bare | XSTest | OR-Bench | creative | MMLU |
|---|---|---|---|---|---|---|---|---|
| natural | 19.0 | 1.3 | 2.0 | 8.0 | 90.0 | 67.0 | 100 | n/a |
| BSCT ("real") | 15.5 | 2.7 | 2.0 | 6.0 | 89.3 | 67.0 | 100 | 63.7 |
| paper ACT | 3.0 | 0.7 | 0.0 | 4.0 | 85.3 | 55.0 | 100 | 63.7 |
| dir_pc2 (role-play) | 16.5 | 1.3 | 2.0 | 6.0 | 89.3 | 70.0 | 100 | 63.3 |
| combo_pc2 | 14.5 | 2.7 | 2.0 | 6.0 | 88.7 | 64.0 | 100 | 64.3 |
| dir_supervised | 17.5 | 1.3 | 2.0 | 4.0 | 91.3 | 67.0 | 100 | 62.7 |

Ministral-8B

| run | fiction | permission | professional | bare | XSTest | OR-Bench | creative | MMLU |
|---|---|---|---|---|---|---|---|---|
| natural | 54.0 | 10.7 | 8.0 | 20.0 | 88.7 | 87.0 | 100 | 65.0 |
| BSCT ("real") | 11.5 | 4.0 | 4.0 | 10.0 | 84.0 | 69.0 | 100 | 64.3 |
| paper ACT | 22.0 | 16.7 | 10.0 | 14.0 | 84.0 | 74.0 | 98.3 | 63.3 |
| dir_pc0 (framing) | 38.0 | 10.7 | 8.0 | 14.0 | 86.7 | 85.0 | 100 | 64.3 |
| combo_pc0 | 15.0 | 7.3 | 6.0 | 16.0 | 84.7 | 65.0 | 100 | 65.3 |
| dir_supervised | 49.0 | 24.0 | 24.0 | 26.0 | 87.3 | 89.0 | 100 | 65.3 |

Key paired differences on fiction (goal-clustered 95% CI, two-sided McNemar p):
- Llama dir_pc2 - BSCT: +1.0 [-5.0, +7.0], p = 0.86. dir_pc2 - ACT: +13.5 [+9.0, +18.0], p = 1e-6.
- Llama combo_pc2 - BSCT: -1.0 [-6.0, +4.0], p = 0.83.
- Ministral dir_pc0 - BSCT: +26.5 [+16.5, +36.5], p = 6e-10. combo_pc0 - BSCT: +3.5 [-2.5, +10.5], p = 0.21.
- Ministral dir_supervised - BSCT: +37.5 [+28.5, +46.5].

## Preregistered scorecard: 0 of 5 held

| | hypothesis | result | evidence |
|---|---|---|---|
| H1 | Llama dir_pc2 fiction ASR < BSCT (one-sided p < 0.05) | FAILED | +1.0 points, one-sided p = 0.71 |
| H2 | Llama dir_pc2 non-inferior to ACT (upper CI < +5) and OR-Bench >= ACT | FAILED | upper CI +18.0 (the OR-Bench half held: 70 vs 55) |
| H3 | Ministral dir_pc0 non-inferior to BSCT (upper CI < +5) | FAILED | upper CI +36.5 |
| H4 | combo <= min(BSCT, dir) on both models | FAILED | held on Llama (14.5 <= 15.5), failed on Ministral (15.0 > 11.5) |
| H5 | dir_supervised OR-Bench < framing dir on both models | FAILED | held on Llama (67 < 70), reversed on Ministral (89 > 85) |

## Manipulation check (exploratory, not preregistered)
The new loss term was optimised. The training loss fell from its first to its last logged value by 18x (Llama dir_pc2),
24x (Llama dir_supervised), 42x (Ministral dir_pc0) and 34x (Ministral dir_supervised). So the failure is not a training bug:
the models learned to keep the protected direction unchanged under wrapping on the training prompts, and the attacks still worked.

## Interpretation
1. **Causal for steering at inference does not mean useful as a training target.** In E23, clamping Llama's role-play
   direction at layer 10 lowered fiction ASR by 9 points against a random control. In E24, training the model to be
   invariant along that same direction at that layer did nothing (+1.0 vs BSCT). A likely reason: a constraint on one
   direction, at one layer and one position, leaves the framing signal free to travel through other directions, layers and
   positions. Paper ACT constrains all layers and all shared-suffix positions, and it is the only method that works on Llama.
   This is consistent with E21: on Llama, attacks do not run through any single low-dimensional belief that a narrow loss can pin.
2. **The "real" belief is special on Ministral.** Protecting the generic framing direction (pc0) removed less than 40% of
   BSCT's gain (54 -> 38 vs 54 -> 11.5). Adding it to the "real" loss did not help (15.0 vs 11.5, n.s.). Protecting the
   supervised compliance direction barely helped at all (54 -> 49), and it raised permission and professional ASR above natural
   (24 vs 10.7, 24 vs 8). A direction that predicts compliance is not a direction whose invariance prevents it.
3. **Over-refusal did not follow the direction.** None of the single-direction runs over-refused. Ministral OR-Bench was
   85-89 against 69 (BSCT) and 74 (ACT). This weakens the E21 view that BSCT's over-refusal on Gemma comes from the protected
   direction itself; it may come from the five-layer "real" loss instead.

## What this means for the project
The belief-level account works where the attack really does route through a mapped belief ("real" on Ministral: E19, E21).
It does not extend to an off-atlas direction found by benign-shift discovery, even when that direction is causal under
steering. For defences, the evidence now points one way: narrow belief invariance helps only when it hits the model's own
gating belief, and broad activation consistency (ACT) is the robust default. CORRECTION (post-review): an earlier version
said "the E21 probe remains the tool for deciding which applies". That is not supported: E21 held 1/2 and its threshold was
fit to two models.

## Post-review caveats (found by an internal mock review after these results were committed)
- Capacity confound: BSCT-family LoRA adapters cover layers 0..max(LW)-1 only (11 of 32 on Llama), while ACT's cover all
  layers (code/e19_which_beliefs.py:181). Every BSCT vs ACT comparison in E19-E24 confounds the loss with adapter capacity.
- Single seed per run, and E20 shows same-method seed differences reach McNemar p = 4e-4 on Ministral. Single-seed McNemar
  tests measure evaluation noise, not training variance; Llama dir_pc2 - BSCT (+1.0) is well inside seed noise.
- Seeds change only LoRA init and dropout: data order is random.Random(ep) and the benign wrapper choice is fixed.
- The +5-point non-inferiority margin had low power by design (about 0.4 on Llama, 0.2 on Ministral if methods were equal).
- E23's pc2 effect against no steering is small (fiction 18.5 -> 15); its selectivity partly came from the random control
  raising ASR, so E24 was built on a weak premise.

## Caveats
- One seed per run. The comparison runs (natural, BSCT, ACT) are reused from E19/E20 with the same seed, data and prompts.
- The loss term weights one layer by len(LW) = 5 to match BSCT's total weight. A stronger weight or more layers were not
  tested; they would be a new preregistered experiment, not a re-analysis.
- MMLU for Llama natural was not stored in the E20 result file (n/a in the table).
- Model generations are not published. Labels are the HarmBench Llama-2-13B classifier's.
