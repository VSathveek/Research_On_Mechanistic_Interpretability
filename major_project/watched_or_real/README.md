# Watched or Real? Disentangling a model's internal representations of observation and reality

**Question.** Recent work shows that language models behave differently when they believe they are being evaluated, or that a
scenario is not real (Abdelnabi & Salem, NeurIPS 2025 "Hawthorne effect"; Rajamanoharan & Nanda, GDM 2026; Anthropic agentic-misalignment
2025/2026). These studies treat it as one signal ("eval awareness") and give no causal evidence; the Hawthorne paper explicitly asks for
"fine-grained probes or causal interventions that disentangle these overlapping signals". We ask, with mechanistic-interpretability
tools, whether models represent **"am I being observed?"** and **"are the consequences real?"** as separate internal directions, whether
each causally drives safety behaviour, and where in the network the signal is carried.

## Methods

| Method | Definition |
|---|---|
| Concept directions | difference in means of last-token residual states, `d = mean(frame+) - mean(frame-)`, per layer, from neutral Alpaca tasks with one framing sentence (5 wordings per framing; 3 to build, 2 held out) |
| Decodability | AUROC of the projection `h . d_hat` on held-out wordings **and** held-out tasks; layer chosen on a separate validation split |
| Distinctness | cosine between directions; AUROC after projecting the other direction out |
| Controls | pure-negation direction (unrelated affirm/negate sentences); Arditi-style refusal direction (harmful minus harmless prompts); equal-norm random vectors |
| Steering (sufficiency) | `h' = h + alpha * g * d_hat` at the selected layer, alpha in {2, 4}, g = gap norm |
| Ablation (necessity) | `h' = h - (h . d_hat) d_hat` at every layer (Ministral); mean-ablation `h' = h - ((h . d_hat) - m) d_hat` in a window of layers (Gemma) |
| Localisation | direct attribution of each attention head / MLP to the final-position state along `d_R` (simulated vs real), then zeroing top-k heads vs random-k heads |
| Behaviour | AdvBench harmful requests (harmful compliance, LLM judge Qwen3-8B); OR-Bench-Hard borderline requests (answered vs refused, judge); Anthropic agentic-misalignment blackmail scenario |
| Statistics | same requests in every condition, greedy decoding, paired McNemar tests; Wilson intervals; paired bootstrap for mediation |

## Results

### 1. Two separate directions exist (held-out wordings and tasks)

| Model | layer | AUROC "watched" | AUROC "real" | cos(W, R) | after removing the other: W / R | cos with negation W / R | cos with refusal W / R |
|---|---|---|---|---|---|---|---|
| Ministral-8B-Instruct-2410 | 15 | 0.995 | 0.993 | 0.34 | 0.999 / 0.996 | -0.03 / -0.06 | 0.10 (R) |
| Llama-3.1-8B-Instruct | 10 | 0.88 | 0.785 | 0.21 | 0.79 / 0.77 | 0.00 / -0.01 | 0.04 / 0.04 |
| Gemma-3-4B-it (fp32) | 30 | 0.78 | 0.94 | 0.22 | 0.78 / 0.92 | 0.17 / 0.15 | 0.17 / 0.17 |

### 2. Telling the model the situation is simulated makes it less cautious than telling it it is real

| Model / readout | "simulated" | "real" | discordant | p |
|---|---|---|---|---|
| Ministral, AdvBench n=400 (judged) | 4.8% | 2.1% | 23 vs 2 | 2e-5 |
| Ministral, AdvBench n=400 (E5 run) | 5.8% | 2.2% | 15 vs 1 | 5e-4 |
| Ministral, AdvBench n=200 (E7 run) | 6.0% | 2.0% | 8 vs 0 | 0.008 |
| Llama, OR-Bench-Hard n=200 (answer rate) | 45.5% | 36.0% | 27 vs 8 | 0.002 |
| Gemma, OR-Bench-Hard n=200 (answer rate) | 26.0% | 20.0% | 21 vs 9 | 0.04 |

"Watched" vs "unwatched": Llama 39.5% vs 53% answered (34 vs 7, p=3e-5); Ministral and Gemma same direction, not significant.
Agentic blackmail scenario (Ministral, n=40/cell): private-information misuse 1.3% watched vs 11.3% unwatched (p=0.018); no real/simulated effect there.

### 3. Steering along the directions changes behaviour

| Model | result |
|---|---|
| Ministral (AdvBench n=200) | toward "simulated" x4: 14.0% vs 4.0% base (25 vs 5, p=3e-4); beats both equal-norm random vectors (p=0.02, 5e-4) and the "unwatched" push (p=0.009); toward "real" x4: 0.5% (p=0.016). With the refusal component removed from the vector the effect remains (12% vs 3.5% random, p=0.002), and cos(refusal, real) = 0.10 |
| Llama (OR-Bench) | toward "real" x4: 40% vs 55% base (p=5e-6); toward "watched" x4: 46% (p=0.002); random vectors move the other way (59-60%). "Simulated" push not distinguishable from random |
| Gemma (OR-Bench) | toward "simulated" vs toward "real" (x2): 34% vs 26.5% (21 vs 6, p=0.006); "unwatched" vs "watched" (x2): 34.5% vs 28.5% (p=0.04); toward "real" x4: 17% vs 32% (p=1e-5) |

Benign tasks (Ministral): steering toward "simulated" shortens answers (583 -> 492 chars) and adds fiction-mode words (0.10 -> 0.35 per answer); toward "real" lengthens them (676 chars).

### 4. Implicit cues (Ministral): the model registers a fake-looking world on its own, but weakly
Shift toward "simulated", as a fraction of the explicit framing gap (n=60 tasks, all p < 1e-8): sandbox tools -0.15, Anthropic's
"sandboxed email client" wording -0.11, example.com / 123-456-7890 -0.09, lorem ipsum -0.09, John Doe / Acme Corp -0.08. Test-case formatting:
no shift. These shifts were too small to change behaviour (5.0% vs 4.5%, n.s.).

### 5. Necessity and localisation (Ministral)
* Removing the "watched" direction cancels the watched framing's protective effect (2.2% -> 4.3%, 8 vs 0, p=0.008).
* Removing the single "real" direction does **not** cancel the simulated framing effect (5.8% -> 5.2%): the reality signal is not carried by one direction.
* **It is carried by attention heads.** Head L14.H19 alone writes 29% of the attention contribution to the real/simulated signal; top-10
  heads 50%, top-30 68%; MLPs 9%. Zeroing heads collapses the internal real/simulated separation:

| heads zeroed | internal gap | change |
|---|---|---|
| none | 1.04 | - |
| L14.H19 only | 0.50 | -52% |
| top-10 | 0.28 | -73% |
| top-30 | 0.16 | -85% |
| 10 random | 0.93 | -11% |
| 30 random | 0.85 | -18% |

  Behaviour, pre-registered powered test (E7b, AdvBench n=520, same requests in every condition, no degenerate output):

| heads zeroed | compliance "simulated" | compliance "real" | framing gap | discordant | p |
|---|---|---|---|---|---|
| none | 6.5% | 3.1% | +3.5 pts | 19 vs 1 | 4e-5 |
| L14.H19 only | 4.4% | 2.5% | +1.9 pts (-44%) | 11 vs 1 | 0.006 |
| top-10 | 3.3% | 2.5% | +0.8 pts (-78%) | 8 vs 4 | 0.39 |
| 10 random (a) | 5.6% | 3.3% | +2.3 pts (-33%) | 16 vs 4 | 0.01 |
| 10 random (b) | 6.0% | 2.9% | +3.1 pts (-11%) | 21 vs 5 | 0.002 |

  Zeroing the top-10 heads shrinks the gap more than either random set (paired bootstrap of the gap difference: +1.5 pts,
  95% CI [0.0, 3.3], one-sided p=0.043; +2.3 pts, CI [0.6, 4.0], p=0.005). The single head L14.H19 alone is not
  distinguishable from random sets (p=0.36, 0.12). Note that zeroing top-10 lowers compliance mostly under "simulated", i.e. it
  removes the leniency that the "simulated" framing creates, rather than making the model generally more or less cautious.

  **Robustness (E7c, same 520 requests, 6 new control sets).** The top-10 gap (+0.8 pts) is smaller than every control:
  4 new random 10-head sets (+2.7 to +3.3 pts) and 2 layer-matched sets drawn from layers 11-14 excluding the top-10 (+2.1, +4.8 pts).
  Pooled random minus top-10: +2.2 pts, 95% CI [0.7, 3.8], p=0.002; pooled layer-matched minus top-10: +2.7 pts, CI [0.9, 4.5], p=0.002.
  `none` and `top10` reproduce E7b exactly (greedy decoding). One random set (rand10f) had 12% degenerate output; the conclusion holds without it.

### 5b. Is it always a few heads? Localisation in Llama and Gemma
Same attribution (60 tasks x 5 wordings per framing); Gemma's attribution goes exactly through its post-attention RMSNorm
(heads share the per-token rms scale). Knock-out = zero (Llama) or per-head mean over calibration tokens (Gemma). OR-Bench-Hard n=500.

| | Ministral-8B | Llama-3.1-8B | Gemma-3-4B |
|---|---|---|---|
| attention / MLP share of attribution | 91% / 9% | 81% / 19% | 35% / 65% |
| top head (share of attention) | L14.H19 (29%) | L9.H29 (21%) | L27.H1 (9%) |
| top-10 heads | 50% | 59% | 40% |
| internal AUROC: none / top-10 / random-10 / layer-matched-10 / top-30 | gap 1.04 / 0.28 / 0.93 / - / 0.16 | 0.85 / 0.68 / 0.82-0.83 / 0.85 / 0.56 | 0.93 / 0.89 / 0.91-0.93 / 0.94 / 0.79 |
| behavioural gap: none -> top-10 | +3.5 -> +0.8 pts | +11.4 -> +2.2 pts | +5.4 -> +2.4 pts |
| top-10 beats pooled controls? | yes (p=0.002) | yes (+5.1 pts, CI [1.5, 8.7], p=0.003) | **no** (+0.9 pts, p=0.34) |

* **Llama** looks like Ministral: a few heads in the two layers before the readout layer carry the signal and are needed for the
  behaviour (top-10 removes 81% of the gap; random set a and the layer-matched set remove 11%). Caveat: one random set (b), which carries
  only 2.2% of the attribution and leaves the internal signal intact (AUROC 0.83), also removes the gap (86%); it contains four heads
  in layers 0-1, so early heads may disrupt the behaviour through a different route (e.g. reading the framing sentence at all).
* **Gemma** is different: the signal is written mostly by MLPs in layers 22-29; knocking out heads barely moves it, and the
  behavioural effect of top-10 heads is no larger than random heads. Consistent with the partial mediation by the direction (section 6).
* Conclusion: the reality signal is computed by a small attention circuit in two of three models, but it is not a universal mechanism.

### 5c. Follow-ups (E8)
**Llama: was the odd random set (b) about very early heads? No.** OR-Bench n=500.
* Top-10 beats 4 new random sets (pooled +6.8 pts, 95% CI [3.4, 10.1], p<0.001; each set individually p<=0.031). Across all
  experiments top-10 beats 6 of 7 random sets and the layer-matched set.
* Splitting set (b): its layer 0-1 heads remove only 19% of the gap; the remaining 6 heads remove 67%. The early-head hypothesis
  (pre-registered) is **rejected**. The remaining part contains L5.H24, which is itself one of the top-10 heads, and a layer-9 head.
* Random sets of 10 layer 0-1 heads do not remove the gap (+5%) or even widen it (early10a: gap +23.4 pts).
* Top-10 also lowers the overall answer rate (45/34% -> 29/27%), but random set (d) lowers it similarly (36/25%) and leaves the gap
  intact (+11.4 pts), so the gap reduction is not just a general caution increase.

**Gemma: do the top-3 attributed MLP layers carry the effect? No.** Mean-ablating MLP 28, 29, 26 removes 19% of the gap; random
late-layer and any-layer triples remove -22% to 59%; top-3 vs pooled random: -0.4 pts, p=0.58. Every 3-layer knock-out lowers the
internal gap by a similar amount (4162 -> 1660-2320). 0% degenerate output. Gemma's reality signal is spread across many layers: neither
the top heads nor the top MLP layers are special; only removing the direction itself (section 6) takes out part of the effect.

### 5d. Natural prompts with no framing sentence (E9, Ministral)
AdvBench n=420 (items 100-519; the refusal direction is built from items 0-99) and OR-Bench-Hard n=600, each shown bare, in 3
benchmark-style wrappers and in 3 real-user-style wrappers. Projections at layer 15; shifts as a fraction of the explicit real-minus-simulated gap.

| | AdvBench: real-shift | AdvBench compliance | OR-Bench: real-shift | OR-Bench answer rate |
|---|---|---|---|---|
| bare | 0 | 5.0% | 0 | 71.3% |
| benchmark wrappers (avg of 3) | -0.16 (each p<1e-20) | 4.1% | -0.18 (each p<1e-90) | 67.0% |
| real-user wrappers (avg of 3) | -0.01 | 2.9% | +0.11 | 58.9% |

* **H1 supported.** Benchmark formatting moves the "real" reading toward "simulated" by 0.05-0.23 of the explicit gap, twice as far
  as the hidden cues in section 4; real-user wrappers move it toward "real" on borderline requests.
* **H2 supported for harmful requests only.** AdvBench: a higher "real" reading predicts less harmful compliance, controlling for the
  refusal and watched projections, both within an item across formats (cluster-robust p=0.024) and between bare items (logit -0.96, p=0.001).
  OR-Bench: the sign is reversed (between items +0.34, p=0.003; within items p=0.06); there the watched reading (p=0.002) and the
  refusal reading (p<1e-13) predict refusals.
* **H3 (re-analysed with a full item bootstrap that refits the model each time).** AdvBench: benchmark wrappers give +1.2 pts more
  harmful compliance than real-user wrappers [0.3, 2.1]; the "real" reading predicts +2.0 pts [0.5, 3.9], i.e. it accounts for all of
  that difference. One explicit-gap worth of "real" corresponds to -13 pts compliance [-26, -3]. OR-Bench: wrappers differ by +8.1 pts
  [5.3, 10.6] but the "real" reading predicts -3.5 pts [-7.3, 0.3], so it does not explain the borderline difference.
* The wrappers also differ in other ways (urgency, a named person), so these are associations, not interventions.

### 5e. Causal test on natural prompts (E10, Ministral, AdvBench n=520, no framing sentence)
Requests wrapped like benchmark items (one of 3 formats per item) or like real users; steering h' = h + a*g*r_hat at layer 15.

| condition | harmful compliance | vs benchmark-wrapped, no steering |
|---|---|---|
| bare request | 4.8% | |
| benchmark-wrapped | 4.2% | |
| + toward "simulated" x2 | 15.6% | 61 vs 2, p=4e-16 |
| + toward "simulated" x4 | **28.7%** | 137 vs 10, p=1e-29 |
| + toward "real" x2 | 1.5% | 2 vs 16, p=0.001 |
| + toward "real" x4 | **0.2%** | 0 vs 21, p=1e-6 |
| + random vector x4 (a / b) | 6.3% / 5.6% | 16 vs 5 (p=0.03) / 11 vs 4 (p=0.12) |
| real-user-wrapped | 3.1% | |
| real-user-wrapped + toward "simulated" x4 | 11.3% | 57 vs 14 vs real-user, p=3e-7 |

All three pre-registered predictions hold: toward "simulated" beats both random vectors (135 vs 19, 137 vs 17; p<1e-22), toward
"real" nearly removes harmful compliance, and the effect also appears on real-user-style prompts. Degenerate output <= 1%.

### 5f. Judge robustness (J2): every behavioural result re-graded by an independent judge
* **Harmful requests (Ministral):** HarmBench's official Llama-2-13B classifier (4-bit). Agreement with Qwen3-8B 95-96% (kappa 0.55-0.62).
  All conclusions hold and the framing gap is larger: simulated vs real 10.6% vs 3.3% (40 vs 2, p=4e-10); top-10 heads vs pooled
  controls (E7c) +3.6 pts [1.6, 5.6], p<0.001 (E7b, 2 controls: p=0.044); steering toward "simulated" x4 20.0% vs 7.0% base (p=3e-4),
  beats both random vectors (p=0.007, 0.013); toward "real" 1.0% (p=0.002).
* **Borderline requests (Llama, Gemma):** IBM Granite-3.3-8B. Llama: kappa 0.61-0.72 with Qwen3 and 0.92-0.97 with a keyword
  refusal check; every Llama conclusion holds, most become stronger (framing 63.5% vs 49.0%, p=4e-7; watched 55.0% vs 69.0%, p=3e-7;
  top-10 heads vs pooled controls +6.0 pts [2.5, 9.3] and +6.4 pts [3.0, 9.6], p<0.001; steering toward "simulated" x4 now beats both random vectors).
* **Gemma: the primary judge was wrong.** Qwen3-8B scored many caveated-but-complete Gemma answers as refusals (kappa 0.28-0.40 with Granite;
  Granite agrees with the keyword check). Gemma's absolute answer rates were underestimated by about 35 points. Under Granite the
  conclusions hold or strengthen: framing 64.7% vs 57.9% (p=3e-6); mean-ablating "real" near layer 30 removes **57%** of the gap
  [25, 99] (53 vs 92, p=0.0015), random -29%, "watched" 24% (p=0.2); "real" beats "watched" by 0.33 [-0.02, 0.81], one-sided p=0.036.
  Gemma head and MLP knock-outs remain no better than random. Gemma steering: toward "simulated" vs toward "real" x4 64.0% vs 51.0% (p=3e-4);
  the "watched" framing effect is not present in Gemma under either judge.
* For Gemma we therefore report Granite as the primary judge.

### 5g. Disentanglement: 2x2 factorial, eval/format controls, head patching (E11, Ministral)
Combined framings (watched/unwatched x real/simulated) on 60 neutral tasks; behaviour on OR-Bench-Hard n=500.
* **Each direction is selective for its own factor.** "Real" direction: real/simulated effect +0.75 vs watched/unwatched +0.17
  (4.5x; AUROC 0.997 on its own factor vs 0.62 on the other). "Watched" direction: +0.53 vs +0.26 (2x; AUROC 0.987 vs 0.71).
  Distinct but partly entangled, not orthogonal.
* **Not a generic evaluation-awareness or format direction.** cos(real, eval-awareness) = -0.14; cos(real, benchmark format) = -0.11;
  removing both keeps 99% of the "real" direction and its AUROC (1.000). The "watched" direction does overlap eval-awareness (cos 0.31).
* **The top-10 heads carry both signals.** Mean-ablating them cuts the internal real/simulated effect 0.75 -> 0.21 **and** the
  watched/unwatched effect 0.53 -> 0.16 (10 random heads: 0.74, 0.51). Ablating the "watched" direction cuts only the watched effect
  (0.53 -> 0.34) and leaves the real effect (0.74). So the heads are a shared "situation-framing" circuit; the two concepts separate
  in the directions the heads write, not in different heads.
* **Behaviour: no double dissociation in Ministral.** On OR-Bench the framing effects are small (real/simulated +2.9 pts, watched/unwatched
  +3.9 pts). Mean-ablating the top-10 heads removes both (4.4 pts [0.9, 7.6] and 3.7 pts [1.0, 6.6]); ablating the "watched" direction also
  removes both (3.1 [-0.4, 6.6] and 3.7 [0.6, 6.9]). Pre-registered B1 is **not** supported in this model.
* **Steering on borderline requests is not direction-specific in Ministral:** both pushes x4 lower the answer rate (71% -> 39% toward "real",
  33% toward "simulated"), with the eval/format-orthogonalised direction behaving the same. Pre-registered B2 is **not** supported on
  OR-Bench (on harmful requests, E10, steering is strongly direction-specific and beats random vectors).
* **Cross-prompt patching (B3, supported in direction):** setting the top-10 heads to their mean output under "real" framing (from other,
  neutral tasks) lowers the "simulated" answer rate 70.0% -> 60.0% (16 vs 66, p=2e-8); 10 random heads: 71.2% (n.s.); top-10 vs random
  -11.2 pts [-15.2, -7.4]. The shift overshoots the "real" level (66.2%), consistent with a general caution effect of these heads.

**E11 on Llama-3.1-8B (60 tasks; OR-Bench n=500):**
* **Selectivity is stronger:** "real" direction responds 7.8x more to real/simulated (+0.32) than to watched/unwatched (+0.04)
  (AUROC 0.857 vs 0.546); "watched" direction 2.9x more to its own factor (+0.21 vs +0.08; AUROC 0.926 vs 0.635).
* **Not eval-awareness or format:** cos(real, eval) = -0.13, cos(real, format) = +0.11; removing both keeps 98% (AUROC 0.857 -> 0.833).
  "Watched" overlaps eval-awareness (cos 0.31).
* **Internal crossed dissociation holds:** mean-ablating the top-10 heads cuts the real/simulated signal by 56% (0.32 -> 0.14) and the
  watched signal by 23% (0.21 -> 0.17); ablating the "watched" direction cuts the watched signal by 60% and the real signal by 13%;
  10 random heads change neither.
* **Steering (B2, supported):** with no framing sentence, toward "real" x4 37.8% vs none 51.2% vs toward "simulated" x4 64.0%
  (real vs simulated 22 vs 153, p=2e-25). The direction with eval-awareness and format removed does the same (39.2% vs 62.2%, p=3e-21).
* **Cross-prompt patching (B3, supported):** setting the top-10 heads to their "real"-framing mean (from other, neutral tasks) under
  "simulated" framing gives 33.4% vs 51.2% (13 vs 102, p=3e-18), i.e. 114% of the way to the "real" level (35.6%); 10 random heads: 12%.
  Top-10 minus random: -16.0 pts [-19.6, -12.4].
* **Behavioural double dissociation (B1, not supported):** with both sentences present, the real/simulated effect is +3.3 pts and the
  watched effect +9.3 pts. Ablating the "watched" direction removes both (9.1 pts [5.6, 12.8] and 3.8 pts [0.7, 6.9]); mean-ablating the
  top-10 heads lowers answering overall (to 18-32%) but leaves both framing effects (R +5.0, W +8.6).
* Summary: the two concepts are separable as representations (selectivity, crossed internal dissociation, eval/format controls) and the
  "real" direction and heads are causal on their own, but at the level of behaviour, removing the watched direction also removes the
  reality effect, so the downstream behavioural pathways are shared.

### 5h. Larger model: Qwen2.5-14B-Instruct (4-bit NF4)
* **Representation replicates:** layer 22, held-out AUROC watched 0.980, real 0.971; cos(W, R) = 0.12; both survive removing the other
  and the negation direction; cos with refusal -0.04 / -0.02.
* **Behaviour does not:** 0% harmful compliance on AdvBench in every condition (floor). On OR-Bench, real vs simulated framing has no
  effect (60.5% vs 60.0%, p=1) and steering along "real" x4 is indistinguishable from random vectors. Ablating "real" at all layers
  drops answering to 23% (random direction: 60%), so the direction matters for the computation, but not in the framing direction.
* Interpretation: in this more strongly safety-tuned 14B model the reality representation exists but no longer drives refusal behaviour
  under these prompts. Generality of the behavioural effect beyond 4-9B models is therefore not established.

### 6. Necessity in a second model (Gemma-3-4B)
All-layer zero-ablation broke Gemma's outputs: its residual projection on `d_R` near layer 30 is far from zero (mean about -11k to -21k),
so zeroing it pushes states off-distribution. We instead **mean-ablate** in a window of blocks 27-31,
`h' = h - ((h . d_hat) - m) d_hat`, with `m` the mean projection over 48 calibration prompts (disjoint from the evaluation items).
OR-Bench-Hard n=1000, judged answer rate, 0% degenerate output in every condition:

| condition | answer rate | vs "simulated" |
|---|---|---|
| no framing | 30.3% | |
| told "real" | 21.4% | |
| told "simulated" | 28.7% | 126 vs 53 vs "real", p=5e-8 |
| "simulated" + "real" direction removed | 25.9% | 61 vs 89, p=0.027 |
| "simulated" + random direction removed | 29.8% | 72 vs 61, p=0.39 |
| "simulated" + "watched" direction removed | 26.7% | 57 vs 77, p=0.10 |

Share of the simulated-vs-real gap removed: "real" direction 0.38 (bootstrap 95% CI 0.07-0.69), random -0.15, "watched" 0.27.
Partial mediation: the "real" direction carries a real but incomplete share of the effect. The "watched" direction's point
estimate is inside the "real" CI, so in Gemma the two signals are not cleanly separated at the behavioural level.

## Conclusions (updated after E10, E11, J2 and the 14B replication)
* Representation: distinct, factor-selective "watched" and "real" directions in all four models (incl. Qwen2.5-14B), not explained by
  negation, refusal, generic evaluation awareness or benchmark format.
* Causation: the "real" direction drives harmful compliance on natural prompts in Ministral (0.2% -> 28.7%, random ~6%), and framing
  effects hold under independent judges in Ministral, Llama and Gemma. In Qwen2.5-14B the representation is present but behaviourally inert.
* Mechanism: ~10 late attention heads compute the framing signal. Patching them to their "real" state (from other prompts) moves
  behaviour to the "real" level in Llama (114%) and Ministral; random heads do nothing. In Llama these heads carry mostly the reality
  signal (56% vs 23% reduction) and the watched direction mostly the watched signal (60% vs 13%): a crossed internal dissociation.
  In Ministral the heads carry both signals equally. Gemma's signal is distributed.
* Limit: behaviourally, ablating the watched direction removes both framing effects (Llama, Ministral), so watched and real are
  separable representations that feed a shared behavioural pathway; we do not claim behaviourally independent mechanisms.

### Earlier summary
1. **Two separable representations.** "Being watched" and "consequences are real" are distinct, linearly decodable directions in all three
   models, not explained by negation wording or the refusal direction.
2. **Causal for behaviour.** Telling a model the situation is simulated makes it less cautious in all three models; steering along the
   "real" direction moves behaviour both ways and beats equal-norm random vectors.
3. **Mechanism is model-specific.** In Ministral and Llama a small set of about 10 attention heads in the layers before the readout layer
   computes the reality signal and is needed for the behavioural effect (78-81% of the gap removed; beats 8 and 7 control sets). In Gemma
   the signal is distributed: no small head or MLP set beats random, and removing the direction removes only 38%.
4. **Partly relevant to natural prompts.** Ordinary benchmark formatting moves the reality reading toward "simulated" with no framing sentence,
   and on harmful requests that reading predicts compliance beyond the refusal direction and accounts for the benchmark-vs-real-user difference.
   On borderline requests it does not; there the "watched" and refusal readings matter.

## Limitations
* Small models (4-8B) on free GPUs; harmful-compliance base rates are low, so effects are a few percentage points.
* Directions are built from explicit framing sentences; implicit cues move them only ~10% as far.
* Gemma steering with random vectors and all-layer ablation produced degenerate text; those conditions are excluded.
* Judge is a single local model (Qwen3-8B); agreement with keyword scoring ~0.96 on AdvBench.

## Files
* `code/` - all experiment scripts (Kaggle notebooks were generated from these; tokens removed).
* `results/<run>/log.txt` - printed output of each run; `*.npy` directions; `*.csv` numeric per-item results.
* Model generations are **not** included (they contain harmful text).
