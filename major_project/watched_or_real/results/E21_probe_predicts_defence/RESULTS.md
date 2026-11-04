# E21 results against the preregistered predictions (see PREDICTIONS.md, committed before these runs)

## Qwen2.5-7B: prediction HOLDS
Probe +3.5 < T=10, so we predicted that BSCT does not beat paper-faithful ACT on fiction attacks.
- Fiction ASR: none 11.2 | BSCT 6.0 | ACT 3.8. BSCT - ACT = +2.2 [-0.8, +5.2], two-sided McNemar p=0.19; one-sided p(BSCT < ACT)=0.95.
- Permission 5.0 | 2.3 | 3.7; XSTest 92.7 | 90.7 | 88.7; OR-Bench 77 | 70 | 70; MMLU 72.7 | 72.0 | 73.0.
- Both defences work and cost little; neither is significantly better, as predicted.

## Gemma-3-4B: first ACT run INVALID, re-run pending
- Paper ACT loss was NaN from the first logged step (MMLU 21%, chance level).
- Cause: ACT targets were cached in fp16, and Gemma-3's residual stream exceeds the fp16 range (inf - inf = NaN).
- Fix: cache in the model's compute dtype (fp32 on Gemma; unchanged for every other model). This is an
  implementation bug fix made before the ACT result could be read, not a choice informed by the outcome.
  Only gemma act_paper is re-run; the v1 files are kept with the suffix _v1_nan.
- Gemma BSCT (valid): every attack goes to 0, but the model over-refuses heavily.
  OR-Bench answer rate 68 -> 16, XSTest 90.7 -> 77.3. MMLU is unchanged (58.7 vs 58.3).
