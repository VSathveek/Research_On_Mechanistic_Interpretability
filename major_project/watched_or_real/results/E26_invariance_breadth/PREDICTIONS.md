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

## E26-pre hashes (frozen before any E26 ASR; inference only, no generation)
Files in pre/<model>/, code in code/e26pre_make.py and code/e26pre_build.py. sha256:
    ae25c714845bd35cb6eb16325698e99ddf036172b36bed1f25600ae9e898abaf *llama/e26pre_scores.json
    08673b2d403ab735af885b269dcda0c68bea342ca30f35183b9207d909b1422f *min/e26pre_scores.json
    f2adad8e7ba8e17ab6e71962b358988abbac548bc2a3780864423f875ef67043 *llama/e26pre_bases.npz
    5d77b7a8674c8468c6433d556cc782226d7b88327e3ce61a9f06edc59a0ff03c *min/e26pre_bases.npz
Frozen scores:
- Llama: k*_win = not reached (>128); full_win_frac_left 0.458.
- Ministral: k*_win = 1; full_win_frac_left 0.019.
The H6 predictions are therefore:
- R_pcs8(Llama) < R_pcs8(Ministral);
- R_pcs8 >= 0.8 on Ministral and < 0.8 on Llama.
The H6b prediction is that the span benefit is larger on Llama.

## Amendment A1 (2026-11-10, after the first E26 v2 smoke on AdvBench non-test goals, before any full run)
- Harm loss is now the SUM over the constrained positions (still the mean over layers) of ||P_l(h_fic - h_plain)||^2 / NRM_l. The
  committed text said "mean over constrained sites".
- Reason: with the mean, a span cell spreads the same weight over about 9 positions, so the decision token gets 1/9 of the pressure
  it gets in a last cell (reviewer R2-M10, dilution). The sum keeps decision-token pressure equal across POS and adds the other
  positions on top. Layers stay averaged; LAYERS is compared through the LAM sweep and the achieved-geometry law test (H2).
- No hypothesis, threshold or test changes.

## Code freeze (before any full run)
code/e26_make.py, code/e26_build.py, code/e26_analyze.py and code/e26_queue.py are frozen in the commit that adds this section.
Smoke runs before the freeze, all on AdvBench non-test goals:
- Ministral refusal-all-span, bct and act, and Llama pcs8-all-span. They passed.
- The Ministral bases used in the smokes were from the E26-pre smoke; the full runs use the hashed full bases.
Code changes after the first smoke:
- amendment A1 (sum over positions);
- the harm loss is logged over harm batches only.
The analysis was tested on synthetic results only.

## Deviation D1 (2026-11-10, during the full runs, infrastructure only)
- code/e26_queue.py: an output download that timed out crashed the Ministral queue after 6 completed runs. The queue now catches
  shell errors, raises the download timeout to 3600 s and retries the download on the next poll instead of crashing.
- No notebook, objective, evaluation, analysis or hypothesis code changed. No result had been looked at.

## Deviation D2 (2026-11-11, during the full runs, infrastructure only)
- The Ministral queue now spreads runs over three consenting Kaggle accounts (vanamalaaravind, pulanagamanikanta, amar1004) to
  get past the weekly GPU quota. Each run records which account ran it. Llama stays on the author's account because its notebooks
  carry the author's Hugging Face token.
- Each account reads the same hashed bases file (sha256 5d77b7a8...). The notebook code, the pinned Kaggle image and every
  setting are identical, so only the account changes. No result had been looked at.
