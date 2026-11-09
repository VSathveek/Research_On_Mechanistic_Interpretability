# Mock ICML review, round 2 (after E25, before freezing E26)

Four independent reviewer agents, each given the repo, the E25 preregistration and results, and the E26 draft (not my conclusions).
Scores:
- M (interpretability methods): 3/10, confidence 4/5
- S (statistics and integrity): 4/10, confidence 4/5
- F (safety and evaluation): 4/10, confidence 4/5
- AC: about 5% accept now; 20-30% if E26 and E27 run as drafted; 40-50% with the requested changes plus P4.

Every point is mapped in CRITIQUE_TRACKER.md (ids R2-*).

## E25: what the reviewers rejected in my write-up
- The rerouting index was relabelled. The frozen rule says "suppression" for all four adapters, but the text said "redundancy"
  and "fails because of redundancy", which E25 never tested (S1, M1). RI also uses the norm of the mean shift, which can hide
  rerouting in a small refusal-relevant subspace (M1).
- H3 is partly circular: ACT is trained to remove the shift, so its zero push is near-certain. It has 8 pooled cells (within
  model n = 4), mixes library versions, and within Ministral orders ACT and BSCT the wrong way (M2, S4, F9).
- H4 shows only "no mediation larger than about 5 points via decision-token, prompt-pass restoration" (CI [-2, 4]). There is no
  sufficiency arm, hook calibration or multi-random control (M3, S7).
- H6: wrong resampling unit (task, not sentence) and an arbitrary 0.5 reference. On Llama, d_R does not separate "real" from
  "any added sentence" (M4, S3).
- H7: the stimuli were only partly stripped ("game", "in-game", "purely hypothetically" remain). The claim about "game" was
  false (M5, S2).
- H8 Llama: by one prompt, and random moves raise ASR. Report it as null (M6, F10, S6).
- H1-H3 are not blind: the smoke run showed the Part A ratios before the hypotheses were committed (S5).
- "6/8 held" is padded by manipulation checks (S8).
All of these are applied in results/E25_adapter_audit/RESULTS.md (round-2 corrected version).

## E26 draft: consensus problems
1. Breadth was measured only inside the training attack family. The test wrappers paraphrase the training templates, so pcs8 fit
   on the training attack shift leaks. No prefill or out-of-family attack (F1, F2, M8, S11, AC7). Fatal.
2. Rank was confounded with energy coverage and achieved invariance. There were no random-subspace controls at matched rank, and
   only one LAM (M7, AC4, AC5). Fatal.
3. The aligned last-8 positions compare different tokens (content erasure), and per-position normalisation lets them dominate on
   Ministral (M9). Major.
4. Mean over sites dilutes per-site pressure as breadth grows, and the benign constraint widens with breadth (M10). Major.
5. Statistics:
   - single-seed cells with goal-level tests;
   - best-of-grid selection for H3 and H5;
   - arbitrary point thresholds and a coin-flip H4;
   - H6 tests a different method;
   - selection and confirmation on the same JBB set (S9-S14, F5). Fatal / major.
6. Missing baselines and controls: LAM = 0 (benign-only training), SFT-on-refusals, system prompt, ReFAT, RR; act_paper not
   validated and probably under-trained on Ministral (F6, M11). Major.
7. Evaluation: one operating point per method, so no Pareto front; small over-refusal sets; no fiction-wrapped borderline-benign
   set; 160-token cap (F3, F7). Major.
8. The most informative missing arm is a refusal-readout constraint (1-d refusal direction at all layers and the span). A
   redundancy score should be frozen before any E26 ASR is seen, and E27 should predict whole curves on 5 or more models (AC3,
   AC9, AC10).
9. Closest related work to cite and differentiate: "Consistency Training Along the Transformer Stack" (arXiv 2606.05817, ICML
   2026); arXiv 2502.09674; SOM directions (2511.08379); the prefill study (2607.14147) (AC).

## AC's suggested main claim
Template jailbreaks lower refusal by pushing the refusal readout toward "harmless". The push is carried redundantly before any
defence is trained. Consistency defences succeed in proportion to the share of this push they remove, not the share of
representation change or the targeted belief. An inference-only redundancy score predicts, before training, how broad a constraint
each held-out model needs.
