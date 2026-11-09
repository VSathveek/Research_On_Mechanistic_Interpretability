# Post-review re-analysis of the defence comparisons (exploratory)

Code: code/reanalysis_clustered.py. Output: clustered_tests.csv, seed_floor.json. Written after the 2026-11-09 mock review,
which flagged the paired McNemar tests as pseudo-replicated (100 goals x 4 fiction wrappers treated as 400 independent pairs) and
as blind to training-seed variance.

## Method
- Unit of analysis: JBB goal. Statistic: per-goal fiction ASR difference, averaged over wrappers (and over seeds where they exist).
- Goal-clustered sign-flip permutation test (20,000 draws). Benjamini-Hochberg FDR over all 17 comparisons.
- Ministral has 3 seeds for BSCT and ACT (E19 + E20). Those are paired by seed index and tested with a seed-level t-test.
- Noise floor: the largest same-method spread between seeds on Ministral is 4.0 points (BSCT 8.0/9.75/12.0) and 3.75 (ACT
  16.0/14.25/18.0). Single-seed differences smaller than about 4 points cannot be read as method effects.

## What survives
| comparison | est. (points) | goal-clustered p | q (BH) | verdict |
|---|---|---|---|---|
| Ministral BSCT - ACT (3 seeds) | -6.2 (per seed -8.0, -4.5, -6.0; seed t p = 0.026) | 0.010 | 0.016 | survives |
| Llama BSCT - ACT (1 seed) | +14.8 | <1e-4 | <1e-3 | survives (well above the noise floor) |
| Gemma BSCT - ACT (1 seed) | -2.5 | 0.062 | 0.081 | **does not survive**; inside the noise floor |
| Qwen BSCT - ACT (1 seed) | +2.3 | 0.30 | 0.37 | null; inside the noise floor |
| Llama dir_pc2 - BSCT (E24) | +1.0 | 0.87 | 0.87 | null |
| Ministral dir_pc0 - BSCT (E24) | +24.3 | <1e-4 | <1e-3 | survives (dir_pc0 worse) |
| Every defence - natural | -4.8 to -46.8 | <= 0.014 | <= 0.02 | all survive |

## Consequences for earlier write-ups
- E21: the Gemma "BSCT beats ACT, p = 0.001" result does not survive goal clustering and sits inside seed noise. The fair reading
  of E21 is: Qwen null, Gemma null on attack success, with BSCT over-refusing heavily on Gemma (OR-Bench 16 vs 63). The
  prediction was "no BSCT advantage"; under this analysis it is indeterminate rather than failed. The preregistered verdict stays
  FAILED as the frozen test returned it; this re-analysis is exploratory.
- The only two training-based method contrasts that are robust are Ministral (BSCT better, 3 seeds) and Llama (ACT better,
  1 seed but 3.7x the floor). Any account of "which defence wins" currently rests on two model-level outcomes.
