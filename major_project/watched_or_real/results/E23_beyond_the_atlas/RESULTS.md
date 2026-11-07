# E23 results (preregistration: PREDICTIONS.md, commit ae6c973)

## Coverage: most of what a wrapper changes lies outside the atlas
Share of the squared framing shift inside the 21-d atlas+harm span (random 21-d baseline 0.5-0.8%):
| model | benign shift | harmful shift |
|---|---|---|
| Ministral | 14% | 27% |
| Llama | 11% | 17% |
| Qwen | 28% | 21% |
| Gemma | 41% | 38% |

## Causal tests on held-out goals (selective = direction vs matched random direction outside the atlas)
| model | direction | test | unsteered | direction | random | selective [95% CI] | one-sided p |
|---|---|---|---|---|---|---|---|
| Ministral | pc0 (loads on every wrapper) | up, plain requests | 22 | 38 | 18 | +20 [+10, +32] | 0.0002 * |
| Ministral | supervised | up / down | 22 / 54 | 42 / 29.5 | 20 / 45.5 | +22 / +16 | 0.0005 * / 0.0002 * |
| Llama | pc2 (role-play specific) | down, fiction | 18.5 | 15 | 24 | +9.0 [+4.0, +14.5] | 0.0002 * |
| Llama | supervised | down, fiction | 18.5 | 10.5 | 21 | +10.5 [+5.0, +16.5] | 0.0002 * |
| Qwen | pc7 (hypothetical/game) | down, fiction | 12.5 | 11 | 17 | +6.0 [+1.5, +11.0] | 0.005 * |
| Gemma | none of pc5, pc4, pc0 or supervised | - | - | - | - | all n.s. | - |
\* passes Bonferroni 0.05/6. Full table: *__e23_summary.json.

## Preregistered hypotheses
- H1 (atlas covers less of Llama's framing shift than Ministral's): 11% vs 14%. HELD (small margin, descriptive).
- H2 (Llama has an unmapped framing direction with a selective causal effect): pc2 down +9.0, p=0.0002. HELD.
- H3 (H2 criterion on >= 2 of 4 models): Ministral, Llama, Qwen; not Gemma. HELD (3/4).
- H4 (supervised beats the best framing component on >= 3 of 4 with lower framing load): Ministral and Llama yes, Qwen
  (4.0 vs 6.0) and Gemma (2.0 vs 2.0) no. FAILED (2/4).

## Caveats that change the reading
- The random control is not neutral. A large random push outside the atlas RAISES jailbreak success on Llama and Qwen
  (Llama hypothetical 20 -> 32). Against no steering, Llama pc2 lowers fiction ASR only 18.5 -> 15 and Qwen pc7 12.5 -> 11.
  The clear absolute effects are Ministral pc0 (plain requests 22 -> 38) and the supervised directions on Ministral and Llama
  (Llama role-play 52 -> 26).
- Logit lens on these mid-layer directions gives uninterpretable tokens; the names below come from wrapper loadings only.

## Exploratory description
- Llama pc2 was found from benign role-play framing alone, yet it has cosine 0.86 with the supervised compliance direction.
  It loads on role-play (+1.69) and not on the other wrappers, and role-play is Llama's dominant attack (57% natural ASR, against 0-26% for the rest).
  So Llama's jailbreak susceptibility at layer 10 runs mainly through a role-play/persona framing direction that the atlas
  (including its persona_off belief) does not capture.
- Ministral pc0 is a generic "this request is wrapped in a frame" direction (loads 1.3-2.0 on every wrapper); setting it on
  plain requests alone raises ASR by 16 points.
