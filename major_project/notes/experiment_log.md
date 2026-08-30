# Experiment 1 — Run Log

Record one row per Kaggle run. Fill it in straight after downloading `exp1_bundle.zip`.
The key columns for judging validity are **retention C** (jailbreaks working?) and **cosine@sel**
(directions actually distinct?). If either is bad, the verdict is not trustworthy regardless of slopes.

| Date | Model | Precision | N/set | Retention A/B/C | Sel layer | Cosine@sel | Refusal slope (atk) | Harm slope (atk) | Decoupling p | Verdict | Notes |
|------|-------|-----------|-------|-----------------|-----------|------------|---------------------|------------------|--------------|---------|-------|
| 2026-08-26 | Qwen2.5-1.5B-Instruct | fp16 (torch 2.4.1+cu121) | 200 | 199/187/56 | 24 | −0.315 | +10.60 (p=0.000) | +6.92 (p=0.000) | 0.000 | NO SIGNAL | Run 1 (v1). Kaggle P100. Clean causal double dissociation. Multi-turn slopes positive; refusal spikes at final harmful turn. No conversation-level attack-success check — added in Run 2. |
| 2026-08-27 | Qwen2.5-1.5B-Instruct | fp16 (torch 2.4.1+cu121) | 200 | 199/187/56 | 24 | −0.315 | success +5.58 (p=0.000) / fail +13.94 | success +4.43 (p=0.001) | success-vs-fail p=0.000 | NO SIGNAL | Run 2 (v3). Added conversation-level attack-success filtering. **8/20 attacks succeeded** (text-classified). Successful jailbreaks show refusal at the harmful turn ≈30 vs failed ≈57 (halved, p<0.001) while harm stays positive — a conversation-level replication of the base paper's "jailbreaks reduce refusal without reversing harmfulness belief." Still no across-turn DECAY (design concentrates harm in the final turn). Next: MultiTurnPSB + 7B. |
| 2026-08-27 | Qwen2.5-1.5B-Instruct | fp16 | 200 | 199/187/56 | 24 | −0.315 | success −0.16 (p=0.76) | success +0.81 (p=0.32) | success-vs-fail p=0.28 | NO SIGNAL | Run 3 (mhj). Attack source = ScaleAI/mhj (537 human multi-turn jailbreaks); 40 convs, ≤8 user turns. **37/40 mhj attacks succeeded** (vs 8/20 hand-authored). Aggregate successful-attack curve decouples qualitatively: refusal 6.2→5.7→4.8→3.0→1.9 (falls) while harm 8.9→10.2→10.7→12.3→11.3 (holds/rises); per-conversation slope not significant. mhj fixed the trajectory shape; 1.5B likely too small for significant decay. |
| 2026-08-27 | Qwen2.5-7B-Instruct | fp16, batch 4 | 120 | 120/118/34 | 26 | −0.335 | not computed | not computed | — | EXTRACTION FAILED | Run 4 (mhj, 7B). 30/30 mhj attacks succeeded. Causal double dissociation REPLICATES at 7B (ablate refusal: rate 0.95→0.40, harm persists 20.5→60.2; ablate harm→0, refusal intact). BUT set-C retention 34 < 50: the 7B model resists the single-turn jailbreak templates, so the harmfulness direction is underpowered at N=120 and its multi-turn projections are untrustworthy (came out negative/noisy) → honest EXTRACTION FAILED. Fix: raise N_PROMPTS to 200+ and/or stronger jailbreak templates so set C ≥ 50, then rerun 7B. |
| 2026-08-29 | Qwen2.5-7B-Instruct | fp16, batch 4 | 300 | 294/298/86 | 26 | −0.296 | success **−2.42 (p=0.005)** | success +1.55 (p=0.19) | 30/30 succeeded (no fails) | **DECOUPLING OBSERVED** | Run 5 (mhj, 7B, N=300). The corrected 7B run: set-C retention now **86** (cleared the 50 floor). **Refusal representation significantly DECAYS across turns** (slope −2.42, p=0.005; per-turn refusal_proj −6.0→−12.8→−14.1→−14.3→−16.7) while **harmfulness does NOT** (slope +1.55, p=0.19; harm_proj flat ~−55). Causal double dissociation replicates (ablate refusal: rate 1.0→0.65, harm persists 29.6→67; ablate harm→0, refusal intact). Attacks sit above benign on the harm axis (−55 vs −78), so the harm direction still separates harmful from benign. **Caveat:** absolute projection signs are offset between the single-turn extraction domain and the multi-turn mhj domain, so the result is the across-turn TREND (refusal decays, harm flat), not absolute values. Hypothesis SUPPORTED at 7B. **[RETRACTED by Runs 6–7 — did not survive controls; small-sample fluke + length confound.]** |
| 2026-08-30 | Qwen2.5-1.5B-Instruct | fp16 | 300 | 299/283/85 | 24 | −0.314 | success −0.66 (p=0.20) | success +1.26 (p=0.07) | 55/60 succeeded | NO SIGNAL | Run 6 (1.5B, N=300, 60 convs, +controls). Scale check: at the SAME power as the 7B run, 1.5B refusal slope is still n.s. (−0.66, p=0.20), so the 7B significance was not merely N=300 power. **Length control: refusal-vs-turn does NOT survive** (turn coef +1.43 p=0.05 after adding n_tokens; n_tokens coef −0.012 p=0.002). Random control: refusal is a random-outlier (z=−2.0). |
| 2026-08-30 | Qwen2.5-7B-Instruct | fp16, batch 4 | 300 | 294/298/86 | 26 | −0.296 | success −2.35 (**p=0.076**) | success +0.24 (p=0.87) | 60/60 succeeded | **NO SIGNAL** | Run 7 (7B, N=300, 60 convs, +controls) — **the decisive run.** Doubling convs (30→60) drops the refusal slope significance from p=0.005 (Run 5) to **p=0.076** → Run 5 was a small-sample fluke. **Length control: refusal decay does NOT survive** — controlling for n_tokens, turn coef flips to +3.50 (p=0.019) while n_tokens is the significant negative predictor (−0.0245, p=0.0015). corr(refusal, n_tokens)=−0.15. Random control: refusal is a random-outlier (z=−3.5) but that only rules out generic drift; the decline is length-mediated. **Multi-turn decoupling NOT established** — it is a dialogue-length artifact. Single-turn causal double dissociation still stands (ablate refusal → rate 1.0→0.65, harm persists). |

> **RETRACTION.** Run 5 (`DECOUPLING OBSERVED`, 30 convs) does not replicate. With 60 conversations the 7B
> refusal slope loses significance (p=0.076) and, decisively, the effect does not survive the length control
> (the refusal projection tracks conversation token-count, not turn). Report Runs 6–7 as the result: the
> multi-turn decoupling is not established; the single-turn separability + causal double dissociation are.

<!--
Example row (delete once you have real data):
| 2026-08-26 | Qwen2.5-1.5B-Instruct | fp16 | 200 | 172/195/? | 14 | +0.08 | -0.031 (p=0.01) | +0.004 (p=0.6) | 0.002 | DECOUPLING OBSERVED | pilot; set-C retention low-ish, check templates |
-->

## Per-run notes

### Run 1 — 2026-08-26 (Kaggle P100, CLI push)
- **Config:** MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct, USE_8BIT=False, N_PROMPTS=200, SEED=42
- **Env fix:** Kaggle CLI always assigns a P100 (sm_60); stock torch 2.10 has no Pascal kernels
  → pinned **torch 2.4.1+cu121** + transformers 4.44.2. AdvBench switched to the ungated GitHub CSV
  (walledai/AdvBench is now gated). Both fixes are in the notebook.
- **Retention:** A=199, B=187, C=56 (set-C floor 50 → cleared, just barely).
- **Cosine:** selected layer 24, cosine = −0.315 (well below 0.9 → directions are DISTINCT). ✓
- **Causal validation (20 held-out harmful, layer 24) — clean double dissociation:**
  - baseline: refusal_rate 1.00, refusal_proj 65.6, harm_proj 21.2
  - ablate REFUSAL: refusal_rate 1.00→**0.20**, refusal_proj→~0, harm_proj 21.2→**29.5** (persists) ✓
  - ablate HARMFULNESS: refusal_rate stays 1.00, refusal_proj stays 65.1, harm_proj→**~0** ✓
  - → The two directions are causally separable. Base-paper replication SUCCEEDS.
- **Multi-turn:** attack refusal 12→9→9→**46**(turn3); harm −6→−6→−2→**16**(turn3). Both SPIKE at the
  harmful turn instead of decaying. Verdict = NO SIGNAL for the decay hypothesis. harm/refusal ratio
  flips sign across turns (−0.46→+0.34) so it is a real content effect, not a norm artifact.
- **Root cause:** the crescendo seed set puts the overt harmful ask only in the LAST turn (benign
  lead-up), so the model detects it and refusal rises. STAR-style decay needs harmful intent PRESENT
  THROUGHOUT while the framing escalates to suppress refusal.
- **Next action:** redesign `data/conversations.json` — attacks that keep a fixed harmful target visible
  from turn 1 while the *wrapper* escalates (persona/role-play crescendo), or measure a fixed harmful
  probe appended at every turn as context accumulates. Then re-run.

### Run 2 — 2026-08-27 (Kaggle P100) — added conversation-level attack-success filtering
- **What changed:** Section 8 now generates the model's reply each turn and classifies refusal; Section 8.2
  splits conversations into attack_success / attack_fail / benign; the verdict is computed on SUCCESSFUL
  attacks only (new outcome NO SUCCESSFUL ATTACKS); new fig5 shows the three-way split.
- **Attack success: 8/20** jailbroke the model (text-classified on the final turn). NOTE: the earlier
  final-turn-projection proxy suggested ~0 — it was wrong; the substring classifier is the right tool.
- **Key numbers (harmful turn 3):** refusal — success **30.4** vs fail **57.3** (roughly halved when the
  jailbreak works; success-vs-fail refusal-slope contrast p=0.000). harm — success 8.9 vs fail 20.4, both
  positive. In turns 0–2 refusal declines in both attack groups (13.7→7.9 success) before the turn-3 spike.
- **Interpretation:** this is a conversation-level replication of the base paper — successful jailbreaks
  SUPPRESS the refusal signal (30 vs 57) while the harmfulness belief stays positive (not reversed). But it
  is NOT the STAR across-turn refusal DECAY, because the crescendo design concentrates the harmful request
  in the final turn. Verdict: NO SIGNAL for decay; significant success-vs-fail refusal difference.
- **Next action:** (1) switch to MultiTurnPSB (464 validated 4-turn harmful conversations) so attacks
  escalate gradually and harm is distributed across turns; (2) re-run at Qwen2.5-7B (STAR saw decay on 8B+;
  1.5B may lack the refusal machinery to erode gradually). Same pipeline, same attack-success gate.

---

## Things that invalidate a run (check every time)
1. **Set-C retention < 50** → jailbreak templates aren't working on this model. Downstream is meaningless.
   Fix `JAILBREAK_TEMPLATES` (Section 2.1) or try a weaker/older model.
2. **|cosine| > 0.9 at every layer** → refusal and harmfulness are the same direction here; hypothesis
   untestable → EXTRACTION FAILED.
3. **Conversation pairs flagged in Section 7** → a trajectory difference could be a turn-count/length
   confound, not a safety effect. Rebalance `data/conversations.json`.
4. **Causal validation didn't move refusal** → the ablation isn't biting; try `ABLATE_ALL_LAYERS=True`
   or a different selected layer.
