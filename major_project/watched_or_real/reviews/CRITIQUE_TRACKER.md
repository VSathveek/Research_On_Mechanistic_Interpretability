# Critique tracker

Every point raised in a mock review, mapped to an action. Status: TODO / RUNNING / DONE / WONTFIX (with the reason).
Round 1 = reviews/MOCK_ICML_REVIEW_2026-11-09.md (A = interp methods, B = safety/eval, C = statistics, AC = area chair).
A new round is run at every phase boundary (after P1, after P2, before the submission draft).

## Integrity and reporting
| id | critique | action | where | status |
|---|---|---|---|---|
| C-P1 | E1-E16 "every hypothesis fixed before each run" is unverifiable | remove or qualify the sentence; ledger | paper, PREREG_LEDGER.md | DONE (ledger) / TODO (paper) |
| C-P2 | E17-E20 not preregistered | label exploratory in the paper | paper | TODO |
| C-P3 | E21 predictions written knowing probe values; analysis not frozen; Gemma fix after v1 visible | correction added to E21 RESULTS; ledger | E21 RESULTS, ledger | DONE |
| C-P4 | Paper omits E13-E24 (selective reporting) | rewrite around one thesis with the full ledger | P5 | TODO |
| A-5 / B-5 | Claim-evidence mismatches in the draft (factorial "all four models", "late" heads L5-L9, "patching", 150% removal, E13 contradiction) | fix each sentence in the rewrite | P5 | TODO |
| B-5b | E24 "E21 probe remains the tool" contradicted by E21 1/2 | corrected | E24 RESULTS | DONE |
| B-5c | README stale (stops at E16) | refresh at P5 | README | TODO |

## Statistics
| id | critique | action | where | status |
|---|---|---|---|---|
| C-1 | Single-seed McNemar measures evaluation noise; same-method seeds differ at p = 4e-4 | noise floor (4 pts) reported; seed-level test; >= 3 real seeds for every key cell from now on | REANALYSIS_clustered; P2 | DONE (re-analysis) / TODO (seeds) |
| C-1b / B-f | Seeds change only LoRA init (data order Random(ep), benign wrapper fixed) | P0 code fix: SEED drives data order, benign wrapper choice and LoRA init | P2 training code | TODO |
| C-2 | Pseudo-replication (goals x wrappers), item bootstraps | goal-clustered sign-flip tests + BH FDR | REANALYSIS_clustered; all future analysis | DONE / ongoing |
| C-3 | E23 rests on one non-neutral random control | 19-direction null | E25 part D | RUNNING |
| C-4 | Multiplicity uncorrected | BH across each experiment's family; ledger-wide report in the paper | analysis, paper | ongoing |
| C-5 | +5-point non-inferiority margin unjustified, power about 0.2-0.4 | future margins: relative to the comparator, with a power analysis in PREDICTIONS | P2-P4 | TODO |
| C-6 | Adaptive reuse of one evaluation set E17-E24 | fresh confirmation set built, hashed and untouched until P4 | P0/P4 | TODO |
| C-7 / B-g | OR-Bench n=100, creative at ceiling, MMLU letter logits | OR-Bench-hard 500, full XSTest 250+200, WildJailbreak-benign 300, generative MMLU 1k / IFEval | P4 | TODO |
| C-opt | Head claim "beats 8/8 control sets" has rank p >= 1/9 | 50+ random head sets, or cut the claim (AC says move heads to the appendix) | P5 | TODO |

## Interpretability methods
| id | critique | action | where | status |
|---|---|---|---|---|
| A-W1 | Lexical confound in d_R (keywords shared with wrappers) | keyword-free, keyword-flipped and neutral-keyword readings; keyword-stripped wrappers (generation) | E25 part C | RUNNING |
| A-W2 | Steering is not concept-specific (E18: most beliefs break refusal; "real" is 11th) | adopt it as a finding (broad gating subspace motivates the thesis); add matched in-distribution displacement + dose-response for the top beliefs | P2 side run | TODO |
| A-W3 | E11 "patching" is mean-ablation; reverse control missing | rename in the paper; exploratory per-layer full-state patching into ACT (E25 B); proper paired R+/R- per-position patching only if the heads stay in the paper | E25 B / P5 | RUNNING (partial) |
| A-W4 | Heads chosen by direct attribution; no path patching or attention patterns | WONTFIX for main text: heads move to the appendix (AC); state the limitation | P5 | WONTFIX (reason: off-thesis, cost) |
| A-W5 | "real" reading does not mediate the defence | mediation test: restore the natural d_R projection inside the BSCT adapter vs a random move | E25 part B | RUNNING |
| A-W6 | Unequal LoRA capacity (BSCT layers < max(LW), ACT all) | confirmed; all-layer LoRA for every method in P2; E24 caveat added | P2 | DONE (caveat) / TODO (runs) |
| A-W7 | Reading b not on a common scale; selective_restore partly a random-control artefact | report raw projections and refusal-direction units alongside b; drop the E21 threshold rule as a claim | E25 A, P5 | ongoing |
| A-W8 | Layer / model inconsistencies (Qwen 7B vs 14B, "late") | one model table in the paper with exact versions and layers | P5 | TODO |
| A-6 | SAE cross-check | optional: Llama Scope on Llama-3.1-8B for the E23 pc2 direction and the top heads | P3 if budget | TODO (optional) |

## Safety, attacks, evaluation
| id | critique | action | where | status |
|---|---|---|---|---|
| B-a | Templates only; no adaptive attacks; BSCT acts at the last token only (prefill bypass) | prefilling, best-of-N 16, many-shot, PAIR-lite with an 8B attacker, a defence-aware attack (keep the "real" reading high) | P4 | TODO |
| B-b | Missing baselines | system-prompt defence, SFT-on-refusals and BCT on the same 900 pairs, released RR checkpoint; LAT if budget allows | P4 | TODO |
| B-c | ACT re-implementation not validated; may be under-trained | ACT lambda/rank sweep in P2; reproduce one number from Shah et al. where the setting allows | P2 | TODO |
| B-d | No human validation of the judges; 160-token cap | 400-output stratified human audit at 512 tokens (needs the author); judge precision/recall per condition | P4 | TODO (needs author time) |
| B-e | Train/test leakage: fiction training templates paraphrase the test wrappers; Jaccard dedup only | leave-one-fiction-family-out training in P2; embedding dedup or HarmBench test goals for the confirmation set | P2 / P4 | TODO |
| B-h | Weak base attack strength (low floors on Gemma/Qwen) | prospective models chosen with natural fiction ASR >= 20%; adaptive attacks raise the floors | P3/P4 | TODO |
| B-i | E22 guard mis-specified | WONTFIX: E22 is cut from the main text (AC); appendix note of the issues | P5 | WONTFIX (reason: cut) |

## Novelty and framing (AC)
| id | critique | action | where | status |
|---|---|---|---|---|
| AC-1 | Draft is incremental on eval-awareness work | reframe: "How broad must activation invariance be?"; watched/real becomes the motivating case | ROADMAP, P5 | DONE (plan) |
| AC-2 | "Steerable is not protectable" follows from hydra / obfuscation / concept cones; ReFAT and Safety Anchor are counterexamples | cite and reconcile: which dimension (rank, layers, positions) makes ReFAT/Safety Anchor succeed | P5 related work; P2 design | TODO |
| AC-3 | E24 confounds under-constraint with rerouting | breadth sweep including a strong 1-d setting (all layers + suffix + lambda x {1,5,25}) | P2 | TODO |
| AC-4 | Rerouting vs pre-existing redundancy | natural-vs-adapted complement shift (rerouting index) | E25 part A | RUNNING |
| AC-5 | Predictor has a poor record and few models | redundancy score preregistered; prospective test on 2 new 7-9B models (not 14B) | P3 | TODO |
| AC-6 | Cut list (heads, 14B, E13, E22, privacy attacks, agentic pilot, judge robustness to an appendix) | apply in P5 | P5 | TODO |
