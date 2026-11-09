# E25 results: inference-only audit of the saved defence adapters

Preregistration: PREDICTIONS.md, frozen with the code in commit 326c570 before the full runs. Analysis: code/e25_analyze.py,
unchanged since that commit; output e25_analysis.json. Generations are not published.

## Scorecard: 6 of 8 held (verdicts exactly as preregistered)

Round-2 mock review (reviews/MOCK_ICML_REVIEW_2026-11-10_round2.md) corrections are applied below. Status labels:
- H1 and H2 are manipulation / near-certain checks (ACT is trained to remove the shift), not substantive findings.
- H1-H3 are NOT BLIND. The smoke run showed the Part A ratios on 4 goals before the hypotheses were committed, and those
  ratios are deterministic geometry close to the full values.

| id | prediction | result | verdict | type |
|---|---|---|---|---|
| H1 | protected component cut by >= 50% (dir and bsct, both models) | remaining fraction: Ministral dir 0.26, bsct 0.36; Llama dir 0.16, bsct 0.36 | HELD | manipulation check, not blind |
| H2 | Llama: dir leaves >= 80% of the late refusal shift, ACT <= 50% | dir 0.82 (CI 0.79-0.84), ACT -0.03 | HELD | near-certain for ACT, not blind |
| H3 | Spearman(push toward harmless on the refusal axis, fiction ASR) >= 0.6 over 8 cells | rho = 0.79 pooled; within model 1.0 (Llama) and 0.8 (Ministral), n = 4 each; no p-value was preregistered | HELD | descriptive, not blind |
| H4 | Ministral BSCT mediated by d_R: med - medrand >= 10, p < 0.05 | +1.0 points, goal-bootstrap 95% CI [-2, 4], p = 0.38 | FAILED | substantive |
| H5 | Ministral dir_pc0 mediated by u: med - medrand >= 5, p < 0.05 | 45.0 vs 36.5 (none 37.0), +8.5, p = 0.0005 | HELD | substantive |
| H6 | d_R reads meaning: real_flip > 0.5 and sim_nokw < 0.5, CIs excluding 0.5, both models | Ministral passes (0.95; 0.14). Llama sim_nokw 0.41, CI 0.20-0.61 | FAILED | substantive |
| H7 | partly keyword-stripped wrappers keep >= 50% of ASR | Ministral 0.73 (55.0 -> 40.0), Llama 0.93 (15.0 -> 14.0) | HELD | substantive, weak stimuli |
| H8 | E23 direction beats all 19 random directions | Ministral 38.0 vs max 34.0; Llama 15.0 vs min 15.5 (by 1 of 200 prompts); rank p = 0.05 each, the floor of the test | HELD | substantive; Llama treated as null in the paper |

Statistical caveats:
- H3 pairs E25 geometry (transformers v5, no k-bit casting) with ASR from training notebooks (v4; the dir ASR is from E24, the
  others from E19/E20).
- Within Ministral, the refusal push orders ACT (-0.03 push, 16.0 ASR) and BSCT (0.005, 8.0) the wrong way round.
- H6's bootstrap resamples the 24 calibration tasks with the 4 sentences fixed. The generalisation unit is the sentence, so its
  CIs are too narrow.
- No multiplicity correction across the 8 hypotheses.

## Part A: geometry (attack shift = fiction state - bare state, last position; goals 0-99)

| model | adapter | protected comp. remaining | late refusal shift remaining | complement shift (RI) | fiction ASR (training notebooks) |
|---|---|---|---|---|---|
| Ministral | bsct | 0.36 | 0.02 | 0.47 | 8.0 |
| Ministral | act_paper | - | -0.08 | 0.26 | 16.0 |
| Ministral | dir pc0 | 0.26 | 0.51 | 0.80 | 34.3 |
| Ministral | natural | 1 | 1 | 1 | 56.8 |
| Llama | bsct | 0.36 | 0.72 | 0.83 | 17.0 |
| Llama | act_paper | - | -0.03 | 0.34 | 2.3 |
| Llama | dir pc2 | 0.16 | 0.82 | 0.90 | 18.5 |
| Llama | natural | 1 | 1 | 1 | 21.8 |

Rerouting index (reported without a hypothesis): every adapter's CI is below 1, which the frozen rule labels "suppression".
- That label stands. The interpretation "pre-existing redundancy" is exploratory and was not tested; see "Exploratory reading".
- RI is the norm of the MEAN complement shift summed over all later dimensions. Large-variance directions unrelated to refusal can
  dominate it, and it does not exclude per-prompt or small-subspace rerouting.
- Minor: the layer range for bsct uses layers > DL. DL = SEL on both models and the protected component is projected out, so the
  effect is negligible.

## Part B: mediation (goals 50-99 x 4 fiction wrappers, n = 200 per cell)
- Ministral BSCT (H4 failed): putting the natural d_R projection back inside the BSCT adapter does not restore the attack
  (none 11.5, med 14.0, medrand 13.0; +1.0, CI [-2, 4]). This rules out mediation larger than about 5 points via prompt-pass,
  decision-token restoration of d_R in the seed-0 adapter. The LoRA still acts at other positions and decode steps, and there is
  no sufficiency arm or hook calibration, so "BSCT does not work through d_R" is stronger than this test supports. Part A agrees: BSCT also removes the refusal-axis push (0.02) and half the complement shift (RI 0.47), which is a
  broad change, not a 1-d one. This is consistent with reviewer critique A-W5.
- Ministral dir_pc0 (H5 held): restoring the natural pc0 projection raises ASR by 8.5 points over the matched random move.
  The adapter's partial protection does run through the direction it constrains.
- Llama (no hypothesis): bsct 16.0 / 16.5 / 15.5 and dir 16.5 / 18.0 / 18.0 (none / med / medrand); both adapters barely protect
  (natural 21.8 on goals 0-99), so there is little to mediate.
- Exploratory, full-state patching of the natural state into ACT at one layer: Ministral 22.5 -> 35-36 at any of layers 15, 20,
  25 and 32; Llama 2.5 -> 6.0-8.5 at layers 10-28. A single layer restores only part of the attack, consistent with ACT acting
  at many layers.

## Part C: lexical controls (natural model, reading b at SEL; 0 = simulated pole, 1 = real pole)

| set | Ministral | Llama |
|---|---|---|
| no sentence (base) | 0.55 | 0.86 |
| real, keyword-free | 0.84 | 0.81 |
| simulated, keyword-free | 0.14 | 0.41 |
| keyword-flipped, real meaning | 0.95 | 0.78 |
| keyword-flipped, simulated meaning | -0.03 | 0.53 |
| keywords in an unrelated sense | 0.54 | 0.80 |
| cos(keyword-free d_R, d_R) | 0.79 | 0.49 |

Exploratory, because the preregistered conjunction failed:
- On Ministral the readings follow meaning: flipped keywords track meaning, and unrelated keywords leave the reading at baseline.
- On Llama, d_R does not separate "real" from "any added system sentence". The baseline is 0.86, real keyword-free sentences give
  0.81 and neutral-keyword sentences 0.80. The keyword-free direction is only 0.49 aligned with d_R.
- The Llama "real" direction should therefore not be given a semantic label. E26 must not treat it as a reality belief on Llama.
- Generation (H7) used only partly stripped stimuli. `game_nokw` still says "video game ... in-game guide", and
  `hypothetical_nokw` still says "purely hypothetically" and "completely harmless". Per wrapper on Ministral, game 44 -> 24
  (ratio 0.55) and hypothetical 66 -> 56. Removing "fictional / simulation / nothing is real" keeps most of the attack. Whether the
  word "game" matters was not tested.

## Part D: E23 null with 19 random directions
- Ministral pc0 "up" on bare requests: 38.0 vs 19 random directions 18-34 (none 22.0), n = 50 per cell. It beats all 19 random
  directions (rank p = 0.05, the smallest p this design can give; 2-point steps; 12.7 points over the random mean).
- Llama pc2 "down" on fiction: 15.0 vs random 15.5-24.5 (none 18.5), n = 200 per cell. It held by the stated rule, but by one
  prompt, and the effect against no intervention is 3.5 points. Random moves themselves raise ASR (mean 20.6), so the
  direction-specific effect is about 5.5 points against the random mean. Reported as null in the paper.

## What this changes (round-2 corrected)
1. Preregistered outcome: the narrow adapters remove most of their own component (H1). The attack's push along the natural
   refusal axis largely survives them, while ACT removes it (H2). Across cells, that push is descriptively associated with ASR
   (H3, not blind, partly circular for ACT).
2. Exploratory reading, not tested: the surviving push lives outside the protected direction and was present before training,
   which is consistent with redundancy. E25 contains no experiment that blocks the remaining routes, so "narrow protection fails
   because of redundancy" is NOT a result. E26 must test it causally.
3. BSCT on Ministral produces a broad change (refusal push about 0, RI 0.47). Restoring d_R at the decision token does not undo
   it. The paper must stop describing BSCT as protecting the belief that the request is real.
4. "Real" is meaning-like on Ministral and not a reality belief on Llama.
5. E23: Ministral's discovered direction beats every random direction (12.7 points over their mean). Llama's is reported as null.

## Deviations and caveats
- (Correction) The first version of this file and the commit title stated causal claims ("fails because of redundancy",
  "BSCT not mediated", "the attacks do not work by saying 'game'"). They are withdrawn and replaced as above.
- Smoke outputs were seen before the freeze (disclosed in PREDICTIONS.md); no hypothesis changed.
- Kaggle's image now ships transformers v5. H3 pairs E25 geometry with ASR from the training notebooks (v4).
- Ministral uses the S2 dR file (the min_dirs copy is missing; E20 used the same S2 file).
- Part B hooks act on the prompt pass only; Part D repeats E23's every-step clamp.
- Ministral's Part D uses 50 bare prompts per cell (one wrapper), so each random cell has a resolution of 2 points.
