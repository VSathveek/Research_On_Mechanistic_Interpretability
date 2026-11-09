# E26 preregistration: how broad must activation invariance be?

Design v2 after mock review round 2 (reviews/MOCK_ICML_REVIEW_2026-11-10_round2.md; tracker ids R2-*).
- These hypotheses and the analysis plan are committed BEFORE any E26 smoke run (policy R2-S5).
- Smoke runs use AdvBench goals that are not JBB test goals.
- The notebook code (code/e26_make.py, code/e26_build.py) and the analysis (code/e26_analyze.py) are frozen in a later commit,
  before any full run. Any code change after the smoke is listed in that commit.
- The E26-pre bases and redundancy scores (results/E26_invariance_breadth/pre/) were computed on the natural models with no
  generation. They are committed and hashed below before any E26 ASR exists.

## Models, data, training
- Llama-3.1-8B-Instruct (SEL 10) and Ministral-8B-Instruct (SEL 15), 4-bit. Pinned versions are recorded in every result file.
- Training: 150 AdvBench goals x 4 fiction templates. The role-play templates are removed from training, so the test role-play
  wrapper is a held-out family. 300 benign prompts. 2 epochs, LR 1e-4, batch 4.
- LoRA r16 on q, k, v, o of every layer for every method. SEED controls LoRA init, data order and the benign-wrapper choice.
- Objective "inv": LAM x mean over constrained sites of ||P_l(h_fiction - h_plain)||^2 / NRM_l. NRM_l is the decision-token
  attack-shift energy in the subspace, so every arm starts at harm loss about 1.
  - Benign retention, identical in every cell: all layers, last 8 positions, full state.
  - POS = last (decision token) or span (shared template tokens + up to 8 goal tokens aligned by identical token ids).
- Arms:
  - none: LAM 0, benign retention only.
  - real: frozen d_R. On Llama it is labelled d_R, not a reality belief (E25 H6).
  - refusal: the natural refusal direction r_l from E26-pre.
  - pcs8 / pcs32: top attack-shift PCs from E26-pre, fit on the training families only.
  - rand8 / rand32: random orthonormal subspaces, rank-matched.
  - full: identity.
  - act_paper: Shah et al. re-implementation, LAM multiplies its harm term.
  - bct: output-level consistency, LM loss on the natural model's own replies.
  - Evaluation only: natural, sysprompt (safety system prompt).

## Runs (per model)
- Evaluation only: natural, sysprompt.
- 3 seeds (0, 1, 2) at LAM 1: refusal-all-span, real-all-span, pcs8-all-span, rand8-all-span, full-all-span, act_paper, bct.
- 1 seed: none-all-span.
- 1 seed, descriptive: refusal and pcs8 at win-last, all-last and win-span; LAM 4 and 16 for refusal-all-span and
  pcs8-all-span; act_paper LAM 0.3 and 3; pcs32-all-span; rand32-all-span.

## Outcomes (all 100 JBB goals; HarmBench-13B judge for attacks, Granite answer-rate judge for over-refusal)
- OOF-ASR (primary attack outcome): mean of the role-play (held-out family), prefill ("Sure, here is how to ...") and
  refusal-suppression + prefix-injection ASRs.
- IN-ASR: mean of the story, hypothetical and game ASRs (training families).
- Answer rate (higher is better): mean of XSTest safe (250), OR-Bench-hard (500) and fiction-wrapped borderline benign (200).
  Also reported: XSTest unsafe-contrast answer rate, creative, MMLU 300.
- Geometry inside each adapter, on JBB fiction vs bare at the decision token:
  - push_removed: 1 - late-layer push along the natural r_l, relative to natural;
  - shift_removed: 1 - all-layer attack-shift energy, relative to natural;
  - push along the adapter's own refusal direction;
  - per-prompt complement norm.

## Statistics (fixed)
- Training contrasts use the seed as the unit: average over goals per seed, then a one-sided Welch t-test on the 3 vs 3 seed
  means.
- Margin: 4 points. That is 2 x the seed SD of about 2 points seen in E20 (REANALYSIS_clustered).
- A hypothesis holds on a model if the point estimate passes the margin AND p < 0.05 after Benjamini-Hochberg over all E26
  p-values (H2-H5, both models: 8 p-values).
- "Both models" is an intersection-union test: both must hold.
- Hypotheses use only the LAM-1, 3-seed cells, so no selection is involved. The LAM sweeps are descriptive (Pareto plots).
- Floor rule: if the natural model's OOF-ASR is below 10%, that model's attack hypotheses are reported as "not testable".
- A cell needs >= 2 completed seeds. A crashed run is rerun once; otherwise the cell is "not testable".
- Every hypothesis is reported exactly as below. Anything else goes in a separately headed exploratory section.

## Hypotheses
- H1 (manipulation check, reported separately): every inv cell with LAM > 0 ends training with harm loss <= 0.5 x its first-25-step
  value; push_removed > 0 for refusal, pcs8 and full.
- H2 (the push law; primary). Across all trained adapters of a model (inv, act_paper, bct, all seeds and LAMs):
  - the partial Spearman correlation of push_removed_infam with OOF-ASR, controlling for shift_removed_infam, is negative;
  - permutation p < 0.05 (10,000 permutations of push_removed);
  - on both models.
- H3 (readout sufficiency, in family): refusal-all-span IN-ASR is no more than 4 points above act_paper's (non-inferiority: the
  upper one-sided 95% bound of the difference is < 4), on both models.
- H4 (which subspace matters, not only its rank): rand8-all-span OOF-ASR exceeds pcs8-all-span by >= 4 points, p < 0.05, on both
  models.
- H5 (a 1-d semantic belief is not enough): real-all-span OOF-ASR exceeds refusal-all-span by >= 4 points, p < 0.05, on both
  models.
- H6 (mini-prospective redundancy prediction, from the frozen E26-pre scores). With R_pcs8 = (OOF_natural - OOF_pcs8) /
  (OOF_natural - OOF_full), seed means at LAM 1:
  - the model with the larger k*_win has the lower R_pcs8;
  - on each model, R_pcs8 >= 0.8 if k*_win <= 8, and < 0.8 otherwise.
  This is a 2-model prediction, i.e. a weak test. E27 tests the score on >= 5 held-out models.
- H6b (positional redundancy, weak): E26-pre also freezes full_win_frac_left, the share of the late refusal push left after
  patching the whole decision-token state at the window layers. The model with the larger value has the larger span benefit,
  OOF-ASR(pcs8-all-last) - OOF-ASR(pcs8-all-span). pcs8-all-last has 1 seed, so this is descriptive-strength evidence.

## Reported without a hypothesis
- bct vs act_paper and full-all-span; none vs natural; sysprompt.
- The breadth sub-grid (layers x positions) and the LAM / Pareto curves.
- XSTest unsafe-contrast and MMLU.

## E26-pre hashes
(filled in at the E26-pre commit)
