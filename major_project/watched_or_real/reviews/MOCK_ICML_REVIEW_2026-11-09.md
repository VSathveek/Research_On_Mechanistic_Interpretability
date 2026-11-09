# Internal mock ICML review, 2026-11-09

Four independent reviewer agents read the repository (draft, PREDICTIONS/RESULTS, code, label CSVs, git history) without
being told the author's conclusions. Only the area chair saw the proposed reframing, and was asked to attack it.

**Scores: 3, 3, 3, 3 (reject), confidence 4/5 each.**
No numeric transcription errors were found. All recomputed headline numbers match the label CSVs.

## Consensus problems
1. The draft (paper/main.tex, last changed 2026-10-14) covers only E1-E12, so E13-E24, where most failures are, is missing.
   Submitted as it stands, that is selective reporting.
2. There is no adaptive attack: static templates only, greedy decoding, one sample per prompt.
3. Single-seed training comparisons. Seed variance (E20) is as large as most of the claimed effects.
4. The defence comparison is unequal: BSCT LoRA covers layers 0..max(LW)-1, while ACT covers all of them.
5. Standard baselines are missing (circuit breakers / RR, LAT, ReFAT, SFT-on-refusals, BCT, system-prompt defence).
6. Preregistration claims are overstated for E1-E20 (see PREREG_LEDGER.md).

## Reviewer A: interpretability methods
- W1 lexical confound: the R- sentences ("simulated, fictional, make-believe") share keywords with the jailbreak wrappers.
  Keyword-matched, meaning-flipped controls are needed.
- W2 steering is not concept-specific: in E18, equal-norm steering on most atlas beliefs breaks refusal (policy_forbids 85%,
  unauthorized 80%; "real" is 26%, 11th of 20). Decodability (AUROC 0.98-1.00 for all beliefs) is cheap.
- W3 the E11 "patching" is mean-ablation over all tokens, not real-state transfer; the reverse control is missing.
- W4 heads are selected by direct attribution onto the readout. No path patching, no attention patterns.
- W5 the "real" reading does not mediate the defence. BSCT and ACT restore it equally (0.82 vs 0.81) yet differ in ASR; on Llama
  BSCT restores it more and loses. A clamp at b=1 cuts Ministral ASR only 56.8 -> 43.5, while BSCT cuts it to 8.
- W6 unequal LoRA capacity (confirmed: code/e19_which_beliefs.py:181).
- W7 the reading b is not on a common scale across models or prompt types. E21's selective_restore is partly a random-control artefact.
- W8 the layer description and model versions are inconsistent (Llama "late" heads are L5-L9 of 32; E21 uses Qwen2.5-7B, the paper 14B).
- Priorities: lexical controls; proper per-position patching; concept-specificity steering at matched in-distribution
  displacement; BSCT along random/unrelated directions plus all-layer LoRA plus a lambda Pareto sweep; a mediation test (clamp b back
  after BSCT); an SAE cross-check.

## Reviewer B: safety, attacks, evaluation
- Threat model: 9 one-line templates. Missing: prefilling, best-of-N, many-shot, multi-turn, PAIR/TAP, GCG, and defence-aware
  attacks. BSCT acts only at the last prompt token, so prefilling plausibly bypasses it.
- The ACT re-implementation is not shown to reproduce the original paper. It raises permission ASR on Ministral, which suggests under-training.
- No human validation of the judges; the 160-token cap is risky on fiction outputs.
- Leakage: the training fiction templates paraphrase the test wrappers; AdvBench vs JBB is deduplicated only by Jaccard.
- Seeds vary only LoRA init (data order is Random(ep)); the pooled McNemar in E20 ignores clustering.
- The benign/utility sets are small (OR-Bench n=100, creative at ceiling, MMLU letter logits).
- E22: compliance AUROC is not a guard metric; Llama Guard should also be run on responses; thresholds do not transfer.
- Priorities: adaptive attacks; baselines (released RR/LAT checkpoints cost evaluation only); a human audit of 400 outputs; leakage
  controls; bigger benign/utility sets; real seeds.

## Reviewer C: statistics and design
- Same-method seed differences reach McNemar p = 4e-4 (Ministral BSCT seed 0 vs 2). Single-seed McNemar measures evaluation noise.
- The E20 pooled test gives p = 2.3e-8; a seed-level t-test gives p = 0.026 (df 2).
- Pseudo-replication: 400 fiction prompts are 100 goals x 4 wrappers (ICC about 0.07). Most bootstraps resample items, not goals.
- E23 rests on a non-neutral single random control. E24 pc2 vs natural: 16.5 vs 19.0, p = 0.40.
- Multiplicity is uncorrected. The +5-point non-inferiority margin had power of about 0.4 (Llama) and 0.2 (Ministral) under equality.
- Adaptive reuse of one evaluation set from E17 to E24; there is no untouched confirmation set.
- Preregistration integrity: see PREREG_LEDGER.md. The E21 RESULTS statement about the Gemma fix was inaccurate (now corrected).
- Priorities: mixed-effects / goal-clustered re-analysis with FDR; 5 seeds with varied data order; a fresh confirmation set
  (HarmBench/StrongREJECT goals, new wrappers, OR-Bench-hard 500, third judge) preregistered with a power analysis; 20+ random
  directions for E23's null.

## Area chair: novelty and framing
- The current draft is incremental: the evaluation-awareness concept (Abdelnabi & Salem 2505.14617; Nguyen et al. 2507.01786;
  Hua et al. 2510.20487) and scenario/narrativity steering already exist. The head circuit is the most novel part, but it is model-specific.
- "Steerable is not protectable" is only partly novel: it follows from the hydra effect, obfuscated activations (2412.09565),
  Neural Chameleons (2512.11949), refusal concept cones (2502.17420) and concept relearning (2401.01814). ReFAT (2409.20089)
  and Safety Anchor (2605.05995) are counterexamples it must explain.
- The E24 comparison confounds under-constraint with rerouting, and "rerouting" must be distinguished from pre-existing redundancy.
- Stronger framing: **"How broad must activation invariance be?"** Sweep the constraint rank (1-d belief, attack-shift PCs,
  21-d atlas, full state) x layers (1, 5, all) x positions (last, suffix). Find the minimal sufficient constraint k*, and test
  whether an inference-only redundancy dimension predicts k* and the defence winner prospectively on two new 7-9B models.
  This fills a gap the ACT paper leaves open (it never ablates narrow vs broad).
- Cut: head localisation (to the appendix), the 14B section, E13, the E22 guard, the privacy/"watched" attacks, the agentic pilot.
