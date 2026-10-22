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

## Second check (mid-October 2026, against the final findings)
Nothing found that reports the combination. ICLR 2027 submissions were not yet indexed, so that venue is unchecked.

| Final finding | Verdict | New closest work | Note |
|---|---|---|---|
| F1 factorial watched x real dissociation, selectivity, orthogonal to eval/format/refusal | **Novel** as a mechanistic result; concept partly reported | Rajamanoharan & Nanda, *Models May Behave Worse When Eval Aware* (LessWrong, Jun 2026): in Gemini CoT, "safety eval" and "simulation / no real consequences" framings go with opposite behaviour (CoT classification only, no causal claim). Abdelnabi & Salem label "scenario recognition" separately but train one probe | Cite both as the conceptual distinction; present ours as the mechanistic counterpart |
| F2 steering "real" changes compliance on unframed prompts | **Partly reported** | Peng, Wang, Zou, Hu, *Do LLMs Know Their Vulnerable Scenarios?* (arXiv 2607.23496): diff-in-means "scenario direction" on Llama-3.1-8B and Ministral-3-8B lowers refusal on direct prompts. Chowdhury et al. 2608.30585: roleplay directions mostly align with refusal | Not a headline claim. Ours differs: real-vs-simulated contrast, monitoring and format controls, random baselines, refusal orthogonality |
| F3 ~10-head circuit, ablation ~80%, cross-prompt patching 114%, crossed dissociation in Llama, distributed in Gemma | **Novel** | Head-level safety work for context: *Attention Head Specialization* (2606.28153), *Depth Charge* (2603.05772); Quirke 2606.08292 (head roles fail cross-prompt transfer) | Our cross-prompt patching directly answers the Quirke critique |
| F4 shared behavioural pathway; representation without behaviour at 14B | Shared-pathway ablation **novel**; scale part **partly reported** | Manek 2606.29196; Knecht et al. 2605.05835; Mehta *The Refusal Residue* 2607.13346; Singh & Chaudhary 2609.22119 | Present 14B as one observation, not a scaling law |

Also to address: Noël 2608.13329 (eval-awareness directions depend on the defining prompt set) -> report robustness across wordings;
Niblett et al. 2607.24758 (compliance gaps persist without consequence language).

**Safe wording.** "To our knowledge, the first factorial (monitoring x consequence-reality) dissociation of linear representations,
extending the conceptual distinction of Abdelnabi & Salem (2025) and Rajamanoharan & Nanda (2026)." "Consistent with scenario
directions reducing refusal (Peng et al., 2026), we isolate a reality direction from monitoring and format confounds, with random
baselines." "First localisation of a framing signal to a sparse set of attention heads that survives cross-prompt patching."
**Avoid:** "first steering of fiction/reality affecting compliance"; "representation decoupled from behaviour at scale is new";
"models distinguish real from simulated" without citing the above.

## Third check (21 Oct 2026): fiction jailbreaks and the reality circuit (E14)
* (a) Fiction wrappers lower the reality reading, matched non-fiction wrappers do not: **partly reported**. Closest: Mendoza,
  "Why do fictional jailbreaks work?" (LessWrong, 20 Oct 2026): a narrativity direction from neutral text (Qwen2.5-7B, Gemma-2-9b),
  residual stream only, no heads, no real-vs-simulated contrast, no matched non-fiction control. Peng et al. 2607.23496: scenario
  directions fit on jailbreak pairs, no fiction/non-fiction separation.
* (b) Reality reading predicts jailbreak success beyond the refusal projection: **novel** (nothing found).
* (c) Patching ~10 heads / clamping the direction blocks fiction jailbreaks with a random-head control: **partly reported** as a technique
  (Chowdhury et al. 2608.30585 head replacement on jailbreak-localised heads; Arai et al. ACML 2025; Assistant Axis capping 2601.10387;
  Mendoza direction subtraction). Novel part: components learned only from neutral explicit-framing data, transferred without re-fitting.
* Must compare against: refusal-direction steering, Assistant Axis capping, Mendoza story direction, Chowdhury/Arai head interventions,
  Jailbreak Antidote, Circuit Breakers. Must report cos with story / eval-awareness / refusal directions, over-refusal (XSTest, OR-Bench),
  and effect on non-fiction jailbreaks (selectivity).
* Avoid "hijacking a belief circuit" and "first mechanistic defense"; use "one route by which fictional framing defeats refusal runs
  through a framing representation learned outside the jailbreak setting".
