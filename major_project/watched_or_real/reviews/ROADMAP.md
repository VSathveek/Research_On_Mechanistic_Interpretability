# Roadmap after the mock review (2026-11-09; ICML 2027 deadline projected around 2027-01-22, unconfirmed)

## Thesis
**How broad must activation invariance be?** Template jailbreaks move refusal through a broad subspace of situational
beliefs (E18: most atlas directions break refusal at equal norm). A defence that pins one belief direction works only where
that subspace is effectively one-dimensional (Ministral "real"), and fails where it is redundant (Llama). We measure the
minimal sufficient constraint (rank x layers x positions) for consistency-training defences, and test whether an
inference-only redundancy score predicts it on models not used to build the score.
"Watched vs real" becomes the motivating case study (about one page). "Steerable is not protectable" becomes a corollary.

## Budget
Two Kaggle accounts, about 30 GPU-h/week each, so roughly 400 usable GPU-h by mid-January.

## Phases
| phase | content | GPU-h | addresses |
|---|---|---|---|
| P0 | Record corrections (done); goal-clustered, seed-level, FDR re-analysis of E17-E24; fix training code (LoRA on all layers for every method; seed controls data order and benign wrapper choice); build and hash a fresh, untouched confirmation set | 0 | C: stats, preregistration; A-W6; B: seeds |
| P1 | **E25** (existing adapters, inference only): redundancy vs rerouting (wrapper and compliance decodable outside the protected direction, natural vs adapted, per layer); mediation clamp (push the "real" reading back after BSCT, and check whether ASR returns); lexical controls for d_R; a 20-random-direction null for E23 pc2 | ~15 | AC; A-W1, W5; C: E23 control |
| P2 | **E26 breadth sweep** on Llama and Ministral, all-layer LoRA throughout: 1-d "real" at 5 layers / all layers / all layers + suffix; 21-d atlas; 8 attack-shift PCs; full state at 5 layers; ACT. Then a lambda sweep for the ASR vs over-refusal Pareto front, and 3 real seeds for the key cells | ~150 | AC framing; A-W6; C: seeds |
| P3 | **E27** redundancy score (iterative nullspace projection of the attack effect, per layer), preregistered; prospective prediction of k* and the winner on 2 new 7-9B models | ~60 | AC: prospective test |
| P4 | Confirmation set (HarmBench or StrongREJECT goals, new wrappers, OR-Bench-hard 500, full XSTest, a third judge). Adaptive attacks: prefilling, best-of-N 16, PAIR-lite. Baselines: system-prompt defence, SFT-on-refusals, BCT, released RR checkpoint. Human audit of 400 outputs (needs the author) | ~100 | B: all; C: fresh set |
| P5 | Rewrite the paper around the thesis, with the full ledger (7/17 held) and the cuts the AC lists | 0 | all |

## Falsifiers stated up front
- A strong single-direction loss (all layers, suffix, swept lambda) matches ACT, so the failure was under-constraint, not breadth.
- Framing information outside the protected direction is unchanged from natural, so it is redundancy, not rerouting (this is
  still compatible with the thesis, but changes the mechanism claim).
- The redundancy score fails to rank-predict the narrow-vs-broad gap on held-out models.
