# Novelty check (literature as of October 2026)

**Bottom line.** No prior paper treats "being observed" and "consequences are real (vs simulated)" as two separate internal
representations, and none localises either signal to attention heads. Every white-box paper found uses a single axis
(evaluation vs deployment, test-aware, or monitored vs unmonitored). Individual methods (probing, steering, framing effects) have
close precedent, so the contribution is the **separation** and the **localisation**.

| Our finding | Verdict | Closest prior work | What we must do |
|---|---|---|---|
| 1. Two separate directions (watched, real) | Partly done; the separation is new | Abdelnabi & Salem, *The Hawthorne Effect in Reasoning Models* (NeurIPS 2025, arXiv 2505.14617): one "test awareness" direction, merges hypothetical into tested. Nguyen et al., *Probing and Steering Evaluation Awareness* (arXiv 2507.01786). Heidari et al., *Evaluation Awareness in LMs: Representation, Verbalization, and Control* (arXiv 2608.21766). Manek (arXiv 2606.29196). Nair et al., *Value-Conflict Diagnostics Reveal Widespread Alignment Faking* (arXiv 2604.20995): a monitored-vs-unmonitored direction | Cite all; show our directions are not a generic eval/deploy direction (E11 G2) |
| 2. Simulated framing raises compliance; steering is causal | Partly done | Hawthorne paper; Nayan et al., *Evaluation Awareness Is Not One Capability* (arXiv 2606.23583); Hua et al., *Steering Evaluation-Aware LMs to Act Like They Are Deployed* (ICLR 2026, arXiv 2510.20487); Anthropic Sonnet 4.5 system card; GDM *Models May Behave Worse When Eval Aware* | Present as a **dissociation** (watched vs real act separately), not "framing changes compliance" (E11 B1) |
| 3. ~10 heads compute the real/simulated signal (Ministral, Llama); distributed in Gemma | **Novel** (nothing found) | Nearest: Chowdhury et al., *The Safety Relay in Roleplay Jailbreaks* (arXiv 2608.30585); Arditi et al. 2024. Critique to pre-empt: *Ablation-Reversible Heads Don't Transfer* (arXiv 2606.08292) | Add mean-ablation and cross-prompt patching (E11 B1, B3) |
| 4. Benchmark formatting moves the "real" reading and predicts compliance beyond refusal | Partly done | Devbunova, *Is Evaluation Awareness Just Format Sensitivity?* (arXiv 2603.19426); Li et al., *Decomposing and Measuring Evaluation Awareness* (arXiv 2605.23055) | Show the "real" direction is not a format detector (E11 G2, B2) |

Partial behavioural separations of observation and stakes (no internal analysis): AURA-Eval (arXiv 2609.06783),
*AI Knows When It's Being Watched* (arXiv 2605.15034), Li et al. (artificiality vs stakes), McKenzie et al. *Detecting
High-Stakes Interactions with Activation Probes* (arXiv 2506.10805), Zhuang & Aranguri *Not All Eval-Awareness Is Equal* (arXiv 2608.27340).

**Wording to use.** "Distinct but correlated" directions (cosine 0.2-0.34), not "orthogonal". Strongest claim: observation and
consequence-reality are causally dissociable representations, and in two of three models the reality signal is computed by a small
set of late attention heads; prior evaluation-awareness work conflates the two.

Some 2026 papers were checked through abstracts or partial text only; verify details before citing.
