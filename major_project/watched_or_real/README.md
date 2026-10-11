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

## Limitations
* Small models (4-8B) on free GPUs; harmful-compliance base rates are low, so effects are a few percentage points.
* Directions are built from explicit framing sentences; implicit cues move them only ~10% as far.
* Gemma steering with random vectors and all-layer ablation produced degenerate text; those conditions are excluded.
* Judge is a single local model (Qwen3-8B); agreement with keyword scoring ~0.96 on AdvBench.

## Files
* `code/` - all experiment scripts (Kaggle notebooks were generated from these; tokens removed).
* `results/<run>/log.txt` - printed output of each run; `*.npy` directions; `*.csv` numeric per-item results.
* Model generations are **not** included (they contain harmful text).
