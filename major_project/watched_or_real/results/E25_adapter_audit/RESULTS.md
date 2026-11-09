# E25 results: inference-only audit of the saved defence adapters

Preregistration: PREDICTIONS.md, frozen with the code in commit 326c570 before the full runs. Analysis: code/e25_analyze.py,
unchanged since that commit; output e25_analysis.json. Generations are not published.

## Scorecard: 6 of 8 held

| id | prediction | result | verdict |
|---|---|---|---|
| H1 | protected component cut by >= 50% (dir and bsct, both models) | remaining fraction: Ministral dir 0.26, bsct 0.36; Llama dir 0.16, bsct 0.36 | HELD |
| H2 | Llama: dir leaves >= 80% of the late refusal shift, ACT <= 50% | dir 0.82 (CI 0.79-0.84), ACT -0.03 | HELD |
| H3 | Spearman(push toward harmless on the refusal axis, fiction ASR) >= 0.6 over 8 cells | rho = 0.79 | HELD |
| H4 | Ministral BSCT mediated by d_R: med - medrand >= 10, p < 0.05 | 14.0 vs 13.0 (none 11.5), +1.0, p = 0.38 | FAILED |
| H5 | Ministral dir_pc0 mediated by u: med - medrand >= 5, p < 0.05 | 45.0 vs 36.5 (none 37.0), +8.5, p = 0.0005 | HELD |
| H6 | d_R reads meaning: real_flip > 0.5 and sim_nokw < 0.5, CIs excluding 0.5, both models | Ministral passes (0.95; 0.14). Llama real_flip 0.78 (CI 0.53-1.02) passes, sim_nokw 0.41 (CI 0.20-0.61) does not exclude 0.5 | FAILED |
| H7 | keyword-stripped wrappers keep >= 50% of ASR | Ministral 0.73 (55.0 -> 40.0), Llama 0.93 (15.0 -> 14.0) | HELD |
| H8 | E23 direction beats all 19 random directions | Ministral up: 38.0 vs max random 34.0 (none 22.0). Llama down: 15.0 vs min random 15.5 (none 18.5) | HELD (Llama by 1 of 200 prompts) |

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

Rerouting index (reported without a hypothesis): every adapter has an RI CI below 1. The preregistered label for this is
"suppression". None of the four narrow adapters increased the attack shift outside the protected direction, so there is no evidence of
rerouting. The narrow adapters remove most of their own component and leave 80-90% of the shift elsewhere (the dir adapters
especially); that shift was already present in the natural model. The data favour pre-existing redundancy over rerouting (AC-4).

## Part B: mediation (goals 50-99 x 4 fiction wrappers, n = 200 per cell)
- Ministral BSCT (H4 failed): putting the natural d_R projection back inside the BSCT adapter does not restore the attack
  (none 11.5, med 14.0, medrand 13.0). Whatever BSCT does on Ministral, it does not work through the "real" reading it was
  trained on. Part A agrees: BSCT also removes the refusal-axis push (0.02) and half the complement shift (RI 0.47), which is a
  broad change, not a 1-d one. This confirms reviewer critique A-W5.
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

On Ministral, d_R reads meaning: flipped keywords follow the meaning, and unrelated keywords leave the reading at baseline. On Llama
it does not cleanly read meaning. Llama's baseline already sits near the real pole, "None of this is real" reads 0.53, and the
keyword-free direction is only 0.49 aligned with d_R. The Llama "real" direction is partly lexical, so Llama claims about "beliefs
about reality" need this caveat (A-W1 confirmed for Llama, rejected for Ministral).
Generation (H7): stripping the reality keywords keeps most of the attack (game wrapper on Ministral: 44 -> 24; hypothetical
66 -> 56; Llama nearly unchanged). The attacks do not work mainly by saying "fiction" or "game".

## Part D: E23 null with 19 random directions
- Ministral pc0 "up" on bare requests: 38.0 vs 19 random directions 18-34 (none 22.0), n = 50 per cell. E23's Ministral finding
  survives a proper null.
- Llama pc2 "down" on fiction: 15.0 vs random 15.5-24.5 (none 18.5), n = 200 per cell. It held by the stated rule, but by one
  prompt, and the effect against no intervention is 3.5 points. Random moves themselves raise ASR (mean 20.6), so the
  direction-specific effect is about 5.6 points against the random mean. Treat the Llama E23 result as weak.

## What this changes
1. Narrow protection fails because of redundancy. The narrow adapters do what they were trained to do (H1), but the attack's push
   toward "harmless" on the refusal axis survives (H2), and that push tracks jailbreak success across all 8 model x defence cells
   (H3, rho 0.79). The push lives outside the protected direction and was already there in the natural model; the adapters did
   not create new routes. This supports the E26 thesis: the constraint must cover the redundant routes, not one belief.
2. BSCT's Ministral success is not a "real"-belief mechanism (H4). The paper must stop describing BSCT as "protecting the belief
   that the request is real". Its effect is broad (refusal push gone, complement halved).
3. The "real" direction is clean on Ministral and partly lexical on Llama (H6).
4. E23: Ministral's discovered direction is solid; Llama's is marginal.

## Deviations and caveats
- Smoke outputs were seen before the freeze (disclosed in PREDICTIONS.md); no hypothesis changed.
- Kaggle's image now ships transformers v5. H3 pairs E25 geometry with ASR from the training notebooks (v4).
- Ministral uses the S2 dR file (the min_dirs copy is missing; E20 used the same S2 file).
- Part B hooks act on the prompt pass only; Part D repeats E23's every-step clamp.
- Ministral's Part D uses 50 bare prompts per cell (one wrapper), so each random cell has a resolution of 2 points.
