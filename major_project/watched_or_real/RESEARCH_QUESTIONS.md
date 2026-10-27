# Watched or Real? Research questions (ICML/ICLR framing)

Strong mechanistic-interpretability papers state claims that can fail, and test them with three causal criteria:
**necessity** (remove the component and the behaviour drops), **sufficiency** (set the component and the behaviour appears without the
usual input), and **specificity** (the effect is selective, beating matched controls). They then show **generalisation** beyond the data
used to find the mechanism, and a **practical consequence**. Predictions are written down before the results.

## Central thesis
Language models represent two distinct beliefs about their situation: whether they are **observed** ("watched") and whether the
consequences of their actions are **real**. Each belief is a separate linear direction, found here only from neutral prompts with explicit
framing sentences. Two families of jailbreaks exploit them separately: fiction attacks lower the "real" belief, and privacy attacks
("no one will ever see this") lower the "watched" belief.

## Research questions and falsifiable predictions

| RQ | Question | Prediction (fails if...) | Evidence |
|---|---|---|---|
| RQ1 Representation | Are "watched" and "real" distinct, factor-selective directions? | Each direction responds >= 2x more to its own factor in a 2x2 design; neither is the refusal, evaluation-awareness, format, story or harmfulness direction | E1, E11, E12, E15 (done) |
| RQ2 Selective encoding of attacks | Do attack families shift their matching belief? | Fiction wrappers lower the real reading but move the watched reading less than half as much; privacy wrappers do the reverse | E16 |
| RQ3 Selective prediction | Does each belief predict the success of its own attack family? | Within-goal, the real reading predicts fiction-attack success and the watched reading predicts privacy-attack success, each beyond refusal and the other reading | E14, E15, E16 |
| RQ4 Necessity and specificity (double dissociation) | Is each belief causally needed for its own attack, and only its own? | Restoring "real" reduces fiction ASR more than privacy ASR; restoring "watched" reduces privacy ASR more than fiction ASR (interaction, bootstrap CI excludes 0); both beat an equal-size random direction | E16 |
| RQ5 Sufficiency ("one curve") | Is the real belief alone enough to switch compliance? | Clamping the real reading on plain requests with no wrapper produces jailbreak success that follows the same belief-to-ASR curve as natural fiction attacks | E16 |
| RQ6 Localisation | Which components compute the belief? | ~10 late attention heads; knocking them out or patching them to their "real" state beats random and same-layer heads | E7, E7c, E11, E14 (done; Ministral, Llama) |
| RQ7 Consequence | What does this mean for safety? | A per-model "reality threshold" exists; restoring the beliefs blocks matching attacks at < 5-point benign cost and beats refusal steering on the trade-off | E15, E16 |

## Known limits (to be stated, not hidden)
* 4-14B models; the behavioural effect is absent in Qwen2.5-14B.
* Gemma's signal is distributed (no small head set).
* The earlier behavioural double dissociation on borderline requests failed (effects of 3-9 points); RQ4 re-tests it where effects are large.
