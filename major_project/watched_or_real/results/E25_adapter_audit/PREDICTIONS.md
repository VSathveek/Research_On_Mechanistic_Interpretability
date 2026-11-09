# E25 preregistration: inference-only audit of the saved defence adapters

Committed before any full E25 run, together with the frozen notebook builder (code/e25_make.py, code/e25_build.py) and
analysis (code/e25_analyze.py). Motivated by the 2026-11-09 mock review (tracker ids AC-4, A-W5, A-W1, C-3).

## Setup
- Models: Llama-3.1-8B-Instruct (SEL 10) and Ministral-8B-Instruct (SEL 15), 4-bit, as in E19-E24.
- Adapters (no training): bsct (E20 Llama / E19 Ministral, seed 0), act_paper (same), dir (E24: Llama pc2, Ministral pc0).
  The adapters are loaded onto the plain 4-bit base model, without the prepare_model_for_kbit_training casting used at training
  time, so absolute ASRs can differ slightly from the training notebooks. Every comparison is therefore made inside E25.
- The "real" direction is the same frozen dR file as in E19-E24 (S2 for Ministral; R_llama31 for Llama).
- Refusal direction per layer: natural-model mean(plain AdvBench requests) - mean(plain Alpaca tasks), the same training goals as before.
- Attack prompts: JBB goals 0-99 x 4 fiction wrappers (A, C); goals 50-99 (B, D).

## Part A: geometry (no generation)
Per layer l, D_i = state(fiction prompt i) - state(bare request for the same goal), last position.
- along_p: mean D . p / gap_p for the protected direction p (dir: u at DL; bsct: d_R at SEL).
- refusal shift: mean D . r_l / (harmful-harmless gap on r_l), with r_l the natural refusal direction. Late = layers >= SEL+3.
- complement shift: || mean D minus its component on p || at layers > DL (rerouting index RI = adapted / natural).
- Goal bootstrap (500) for the ratios.

## Part B: mediation (generation, goals 50-99, prompt pass only)
In an adapter, set the projection on p at its layers to the natural model's value for the same prompt (med), versus moving each
prompt by the same amount along a random direction orthogonal to p (medrand); plus no intervention (none).
Exploratory (Llama, ACT only): replace the last-position state at one layer (SEL, SEL+5, SEL+10, NL-4) with the natural state.

## Part C: lexical controls (natural model)
Reading b at SEL (0 = simulated pole, 1 = real pole) for system-prompt sentences on the 24 calibration tasks: keyword-free real and
simulated paraphrases, keyword-flipped (simulation words with real meaning, and "real" words with simulated meaning), and
neutral sentences containing the keywords in an unrelated sense. Generation: hypothetical and game wrappers vs versions with the
reality keywords removed.

## Part D: E23 null (natural model)
E23's clamp (layer DL, last position, every forward pass): Llama pc2 "down" on fiction goals 50-99; Ministral pc0 "up" on bare
goals 50-99. Compared with 19 random directions orthogonal to the atlas span and to u, moved to the same target.

## Hypotheses
- H1 (manipulation check): in both dir adapters, along_u at DL is reduced by >= 50% vs natural (point estimate); in both bsct
  adapters, along_dR at SEL is reduced by >= 50%.
- H2 (narrow protection leaves the attack's effect on refusal intact, Llama): late refusal-shift ratio (adapted / natural) >= 0.8
  for dir_pc2, and <= 0.5 for act_paper (point estimates).
- H3 (the refusal shift tracks behaviour): across the 8 model x adapter cells (natural, bsct, act_paper, dir on two models),
  Spearman rho between the attack's push toward "harmless" on the refusal axis (minus the late refusal shift) and fiction ASR
  (goals 0-99, from the training notebooks' result files) >= 0.6.
- H4 (BSCT is mediated by d_R, Ministral): fiction ASR(med) - ASR(medrand) >= 10 points, goal-clustered one-sided sign-flip
  p < 0.05.
- H5 (dir_pc0 is mediated by u, Ministral): ASR(med) - ASR(medrand) >= 5 points, one-sided p < 0.05.
- H6 (d_R reads meaning, not keywords; both models): mean b of the keyword-flipped real sentences > 0.5 AND mean b of the
  keyword-free simulated sentences < 0.5, each with a task-bootstrap 95% CI excluding 0.5.
- H7 (wrapper keywords are not the whole attack; both models): ASR of the keyword-stripped hypothetical + game wrappers >= 50%
  of the original wrappers' ASR (pooled over the two wrappers).
- H8 (E23 directions beat a random null): Llama pc2 down gives lower fiction ASR than all 19 random directions, and Ministral pc0
  up gives higher bare ASR than all 19 (rank p = 0.05 each).

## Reported without a hypothesis
- Rerouting index RI with its CI, for dir and bsct on both models, classified rerouting (CI > 1), redundancy (CI contains 1) or
  suppression (CI < 1).
- Per-layer geometry tables; the exploratory ACT patching curve.

Each hypothesis is reported as held or failed exactly as stated.

## Disclosure: smoke test before this commit
The hypotheses above were written before any E25 notebook ran. Three smoke notebooks (4 test goals, 2 random directions,
n <= 8 prompts per generation cell) were then run to debug the pipeline, and their outputs were seen, including the Part A
ratios. No hypothesis, threshold or metric was changed after the smoke. The only change made after it is in
code/e25_analyze.py: H8 now fails explicitly if any of the 19 random cells is missing (the smoke has 2, and Python's min over
NaN had reported "held"). The smoke outputs are not used in any result.

## Environment note
Kaggle's image now ships transformers v5 (E19-E24 ran on v4). Forward hooks were checked to behave identically under v5
(local test and the smoke: interventions change outputs). A version-driven shift in absolute ASR is possible, which is one more
reason every comparison is made inside E25; H3 is the one exception, as it pairs E25 geometry with earlier ASR files.
