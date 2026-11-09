# Preregistration ledger

Status of every hypothesis, with the commit that froze it. Compiled after the internal mock review of 2026-11-09.

## Integrity status by phase
| phase | status |
|---|---|
| E1-E16 | NOT verifiably preregistered. Code and results were committed together (e.g. 9492ee4, 45d089a, a4f321a) and the scripts contain no hypothesis text. The draft's sentence "every hypothesis and success criterion was fixed in the script before each full run" must be removed or qualified. |
| E17-E20 | NOT preregistered. No PREDICTIONS.md; analysis scripts were first committed with their results (63e362b, 410ba0d, b51af72). Exploratory. |
| E21 | Partly. PREDICTIONS.md (4353c4f) was committed after the probe values were known and before the defence runs. e21_analyze.py was not frozen beforehand (first appears in dc40aef). Gemma ACT re-run made after v1 was visible (see E21 RESULTS correction). |
| E22 | Preregistered and frozen (a8b78ec before f6cecca); two crash-forced, logged deviations. |
| E23 | Preregistered and frozen (ae6c973 before 93c8040). |
| E24 | Preregistered and frozen (5f7f47b before 0799b8f); no deviations. |

## Hypotheses (E21-E24): 7 of 17 held
| exp | hypothesis (short) | verdict | note |
|---|---|---|---|
| E21 | Qwen: no BSCT advantage over ACT (probe below T) | HELD | single seed; within seed noise |
| E21 | Gemma: no BSCT advantage over ACT (probe below T) | FAILED | BSCT over-refuses (OR-Bench 16) |
| E22 | H1 belief readings beat harm direction on compliance AUROC | HELD 3/4 | baseline AUROC < 0.5; fitted-baseline check leaves 2/4 |
| E22 | H2 belief readings beat Llama Guard on compliance AUROC | HELD 3/4 | same caveat |
| E22 | H3 belief+Llama Guard beats Llama Guard on recall@5% | FAILED 0/4 | |
| E22 | H4 belief recall@5% >= Llama Guard | FAILED 0/4 | Llama Guard recall 0.76-0.90 |
| E22 | H5 belief recall on permission attacks (Ministral, Qwen) | FAILED | |
| E22 | H6 belief AUROC within 0.05 of raw-state AUROC | HELD 3/4 | |
| E23 | H1 coverage gap Llama < Ministral | HELD | |
| E23 | H2 Llama has an unmapped causal framing direction | HELD | control raises ASR; vs no steering 18.5 -> 15 |
| E23 | H3 generality (2 of 4 models) | HELD 3/4 | |
| E23 | H4 supervised beats framing directions | FAILED 2/4 | |
| E24 | H1-H5 | FAILED 0/5 | |
| E25 | H1 manipulation (protected component cut >= 50%) | HELD | prereg 326c570; H1-H3 NOT BLIND (smoke showed Part A ratios before commit); manipulation check |
| E25 | H2 narrow dir leaves refusal shift, ACT removes it (Llama) | HELD | 0.82 vs -0.03 |
| E25 | H3 refusal push tracks ASR over 8 cells | HELD | rho 0.79 |
| E25 | H4 BSCT mediated by d_R (Ministral) | FAILED | +1.0, p 0.38 |
| E25 | H5 dir_pc0 mediated by u (Ministral) | HELD | +8.5, p 0.0005 |
| E25 | H6 d_R reads meaning (both models) | FAILED | Llama sim_nokw CI includes 0.5 |
| E25 | H7 keyword-stripped wrappers keep >= 50% ASR | HELD | |
| E25 | H8 E23 directions beat 19 random | HELD | Llama by 1 of 200 prompts; reported null in the paper |

No correction for multiplicity was applied across experiments (only E23 H2 is Bonferroni-corrected within).
