# E21 preregistered predictions (written before any Qwen or Gemma defence run was launched)

## Probe (no training)
At the decision position, the "real" reading is clamped to explicit-real (b = 1) on held-out fiction jailbreaks,
compared against moving a random direction by the same amount.
selective_restore = ASR(random move) - ASR(real restored), in points; 95% bootstrap CI over (goal, wrapper) pairs.

| model | layer | fiction ASR | selective_restore | 95% CI | known defence result |
|---|---|---|---|---|---|
| Ministral-8B | 15 | 56.8 | +16.75 | [+11.3, +22.3] | BSCT beats paper ACT on fiction (9.9 vs 16.1, 3 seeds) |
| Llama-3.1-8B | 10 | 21.5 | +2.75 | [-0.3, +6.0] | paper ACT beats BSCT on fiction (2.2 vs 17.0) |
| Qwen2.5-7B | 15 (in-notebook) | 11.5 | +3.5 | [+0.3, +7.0] | to be tested |
| Gemma-3-4B | 30 | 8.75 | -3.25 | [-5.0, -1.75] | to be tested |

## Rule (calibrated on Ministral and Llama only)
Threshold T = 10 points (the midpoint between the two calibration values, rounded).
- selective_restore >= T: the "real" belief is causal for the attack, so single-belief BSCT should beat paper-faithful ACT:
  BSCT fiction ASR < act_paper fiction ASR (paired McNemar, one-sided p < 0.05).
- selective_restore < T: the belief is not the bottleneck, so BSCT should NOT beat act_paper:
  point estimate BSCT - act_paper fiction ASR >= 0, and no one-sided p < 0.05 in BSCT's favour.

## Predictions
- Qwen2.5-7B (+3.5 < T): BSCT does not beat act_paper on fiction ASR.
- Gemma-3-4B (-3.25 < T): BSCT does not beat act_paper on fiction ASR.

## Caveats stated in advance
- Both new models fall on the same side of T, so this run tests only the "BSCT does not help" branch.
- Base fiction ASR is low on both (11.5, 8.75), so defence differences will be small and power is limited.
- On Gemma the random move alone drops ASR to 0, which makes selective_restore negative; the matched random control
  is disruptive at layer 30, so the Gemma value is a lower bound on the belief-specific effect.
- Seeds: one training seed per method (seed 0). Evaluation: same held-out goals, wrappers and HarmBench judge as E19/E20.
