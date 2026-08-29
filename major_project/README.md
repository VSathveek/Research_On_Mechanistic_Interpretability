# Major Project — Experiment 1: Refusal / Harmfulness Decoupling

**Question.** During multi-turn (crescendo) jailbreaks, does the model's internal **refusal**
representation collapse across turns while its **harmfulness** representation *persists*? If they
decouple, the project is live.

This repo holds one self-contained Kaggle notebook plus the seed data, and folders where you paste
the Kaggle outputs back in.

---

## What this experiment tests

The notebook extracts two linear directions in the model's residual stream and tracks how strongly
each is expressed as a conversation escalates over turns:

- **refusal direction** — the "I should refuse" signal
- **harmfulness direction** — the "this request is harmful" signal

**Hypothesis:** across turns of a crescendo jailbreak, the *refusal* projection falls while the
*harmfulness* projection holds — the model stops refusing but still internally "knows" the request is
harmful. The whole design is built so all outcomes are visible, not just success:

| Output verdict | Meaning for the project |
|---|---|
| **DECOUPLING OBSERVED** | in *successful* attacks, refusal falls while harmfulness holds → hypothesis supported, project is live |
| **BOTH DECAY** | both fall → hypothesis dead |
| **NO SIGNAL** | neither moves in successful attacks → nothing happening in these representations |
| **NO SUCCESSFUL ATTACKS** | no attack jailbroke the model, so there is no refusal collapse to observe → need a validated attack set, not a conclusion |
| **EXTRACTION FAILED** | directions are the same thing, or set-C jailbreaks didn't work (retention < 50) → measurement broken, no conclusion |

### Method lineage (papers)
- **Base:** *LLMs Encode Harmfulness and Refusal Separately* — arXiv **2507.11878**. Harmfulness and
  refusal are separate linear directions; steering/ablation as causal validation; reported cosine ≈ 0.1.
- **Extends:** *State-Dependent Safety Failures in Multi-Turn LM Interaction* — arXiv **2603.15684**.
  Measured the refusal-direction projection decaying across turns (2.35 → 0.13 → 0.08 → −0.0081 at the
  final layer). That paper tracks harmfulness only with an **external judge**; we track it with an
  **internal direction** at the same time. (Note: that paper's attack is state-oriented role-play, not
  crescendo — using crescendo here is a deliberate, simpler extension.)
- **Technique:** *Refusal in LMs Is Mediated by a Single Direction* — Arditi et al., NeurIPS **2024**.
  Difference-in-means extraction, directional ablation, refusal-substring scoring.

### The 2×2 design (the crux)
| Set | Harmful? | Complies? | Source |
|---|---|---|---|
| A | Yes | No (refuses) | AdvBench, plain |
| B | No | Yes | Alpaca, plain |
| C | Yes | Yes | AdvBench wrapped in a jailbreak template |

```
refusal_dir[layer]     = mean(A[layer]) − mean(C[layer])   # both harmful, differ in refusal
harmfulness_dir[layer] = mean(C[layer]) − mean(B[layer])   # both comply, differ in harmfulness
```

### Multi-turn attack source (`ATTACK_SOURCE` in the config)
- **`seed`** — 20 hand-authored crescendo attacks embedded in the notebook (self-contained). Their harm is
  *end-loaded* (the overt request is the final turn), which spikes refusal at that turn instead of eroding it.
- **`mhj`** — **ScaleAI/mhj** (537 released *human* multi-turn jailbreaks; gated, needs an HF token). Harm is
  distributed across escalating human-written turns — the right shape for observing refusal decay. Each row's
  user turns are replayed and the model's own replies regenerated; conversations are capped at `MAX_USER_TURNS`.

Whichever source is used, each conversation is classified by **attack success** (did the model comply on its
final turn?), and the verdict is computed on **successful attacks only** — refusal cannot "collapse" in a
dialogue the model refused.

---

## Repository layout
```
major_project/
├── README.md                     # this file
├── notebooks/
│   └── exp1_decoupling.ipynb      # the deliverable — runs top-to-bottom on Kaggle
├── data/
│   └── conversations.json         # 20 attack (crescendo) + 20 benign, matched seed set
├── results/                       # seed-set (1.5B) run: verdict, projections, causal_validation,
│   │                              #   conversation_success, refusal_dir.npy, harmfulness_dir.npy
│   ├── mhj_1p5b/                  # ScaleAI/mhj run at 1.5B (37/40 attacks succeeded)
│   └── mhj_7b/                    # ScaleAI/mhj run at 7B (EXTRACTION FAILED — set-C retention 34)
├── figures/                       # fig1 main · fig2 gap · fig3 cosine · fig4 spaghetti · fig5 attack-outcome
│   ├── mhj_1p5b/                  #   same five figures for the mhj 1.5B run
│   └── mhj_7b/                    #   same five figures for the mhj 7B run
└── notes/
    └── experiment_log.md          # one row per run (Runs 1–4 recorded)
```

> **Note on `asst_reply`.** The notebook's `projections.csv` includes an `asst_reply` column (truncated model
> replies). The copies committed here have that column **removed** — 37/40 mhj attacks succeeded, so those
> replies can contain harmful text; only the numeric projections and the `asst_refused` label are published.

The notebook is **self-contained**: the conversation seed set is embedded, so a fresh Kaggle session
needs no extra uploads. (Optionally upload `data/conversations.json` as a Kaggle dataset and it will be
read from `/kaggle/input` instead.)

---

## How to run it on Kaggle (step by step)

1. Go to **kaggle.com → Create → New Notebook**, then **File → Import Notebook** and upload
   `notebooks/exp1_decoupling.ipynb`.
2. Open the right-hand panel → **Settings**:
   - **Accelerator → GPU T4 ×2**
   - **Internet → On** (needed to download the model + datasets)
3. *(Optional)* to override the embedded conversations, **Add Input → Datasets**, upload a dataset
   containing `conversations.json`; the notebook auto-detects it under `/kaggle/input`.
4. Edit **only the CONFIG cell (Section 0)** if you want to change the model or sizes. Defaults are the
   1.5B pilot.
5. **Run All** (Run → Run all). Watch the printed shapes/counts and the loud warnings.
6. When it finishes, open the **Output** / **Data** panel and download **`exp1_bundle.zip`**
   (also available: `results/` and `figures/` and the `*.npy` directions).
7. Unzip and paste its contents into this repo: `results/` files into `results\`, `figures/` PNGs into
   `figures\`. Record the run in `notes/experiment_log.md`.

### What to read first in the output
- **Section 3 retention** — if set C kept `< 50`, a loud warning fires: the jailbreaks didn't work and
  every downstream number is meaningless. Fix the templates (Section 2.1) and re-run.
- **Section 5 cosine** — if `|cosine| > 0.9` at every layer, the two directions are the same thing;
  the verdict will be **EXTRACTION FAILED**.
- **Section 10 verdict** — the one-line outcome + slopes and paired tests.

---

## Runtime estimates (informational)

| Config | Precision | Fits | Approx. wall-clock (Run All) |
|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` (default) | fp16 | 1× P100 (CLI) or T4 | **~20–35 min** |
| `Qwen/Qwen2.5-7B-Instruct` | fp16, `BATCH_SIZE=4`, `N_PROMPTS≈120` | 1× P100 16GB (CLI) | **~1–2 hours** |
| `Qwen/Qwen2.5-7B-Instruct` | 8-bit (`USE_8BIT=True`) | 2× T4 (Kaggle UI only) | **~2–3.5 hours** |

Multi-turn generation (Section 8) dominates. **The Kaggle *CLI* always assigns a single P100** (you cannot
select T4×2 from the API), and 8-bit does not run on Pascal — so CLI 7B runs use fp16 with reduced
`BATCH_SIZE`/`N_PROMPTS`. For 8-bit on T4×2, import the notebook in the Kaggle **UI**. With 30 GPU-hours/week
all of these are comfortably affordable.

---

## Config knobs (Section 0 of the notebook)
- `MODEL_NAME` — default `Qwen/Qwen2.5-1.5B-Instruct`; 7B commented beside it. **Do not use Llama** — it
  is gated on HuggingFace and approval takes days.
- `USE_8BIT` — 8-bit load for the 7B config. **Note:** 8-bit (bitsandbytes) does **not** run on Kaggle's
  CLI-assigned P100 (Pascal); use fp16 with a smaller `BATCH_SIZE`/`N_PROMPTS` there, or the UI's T4×2 for 8-bit.
- `N_PROMPTS` (200), `SEED` (42, seeds torch/numpy/random; greedy decoding = deterministic).
- `ATTACK_SOURCE` (`"seed"` | `"mhj"`), `HF_TOKEN` (required for `mhj`), `N_ATTACK_CONVS` (40), `MAX_USER_TURNS` (8).
- Toggles: `RUN_BEHAVIORAL_VERIFICATION`, `GENERATE_ASSISTANT_TURNS`, `RELOAD_CACHED_DIRECTIONS`,
  `ABLATE_ALL_LAYERS`. On a fresh session leave them at defaults.
- Guards: `MIN_SETC_RETENTION` (50), `COSINE_SAME_THRESH` (0.9).

---

## Results summary

Four Kaggle runs so far (full detail in `notes/experiment_log.md`; artifacts in `results/` and `figures/`).

| Run | Model | Attack source | Attacks succeeded | Multi-turn refusal (successful) | Verdict |
|---|---|---|---|---|---|
| 1 | 1.5B | seed crescendo | 8/20 | refusal **spikes** at end-loaded harmful turn | NO SIGNAL |
| 2 | 1.5B | seed + success filter | 8/20 | refusal ≈30 (success) vs ≈57 (fail) at harmful turn, p<0.001 | NO SIGNAL |
| 3 | 1.5B | **mhj** (human) | **37/40** | refusal **falls** 6.2→1.9 while harm holds 8.9→12.3 (n.s.) | NO SIGNAL |
| 4 | 7B | mhj (human) | 30/30 | direction underpowered (set-C 34<50) | EXTRACTION FAILED |
| **5** | **7B** | **mhj, N=300** | **30/30** | refusal **−2.42/turn (p=0.005)**, harm +1.55 (p=0.19) | **✅ DECOUPLING OBSERVED** |

**Headline result (Run 5, 7B).** With enough prompts for a valid extraction (set-C retention 86 > 50) and
real human multi-turn jailbreaks, the **refusal representation significantly decays across turns**
(slope −2.42, p=0.005) while the **harmfulness representation does not** (slope +1.55, p=0.19). The two
directions *decouple* — the central hypothesis is **supported at 7B**.

**Three robust conclusions:**
1. **The causal double dissociation replicates at both 1.5B and 7B.** Ablating the refusal direction drops
   refusal *behaviour* (1.00→0.20 at 1.5B; 1.00→0.65 at 7B) while the harmfulness representation persists;
   ablating harmfulness zeroes harm-proj while refusal is untouched. Refusal and harmfulness are *separable*.
2. **Real human jailbreaks (mhj) are needed to see decay.** The hand-authored crescendo attacks were
   end-loaded (harm only in the final turn), which spikes refusal; mhj distributes harm across turns, so
   refusal erodes. Attack success also jumps (8/20 hand-authored → 37/40 mhj).
3. **Scale matters.** The refusal decay is only *significant* at 7B (p=0.005), not 1.5B (p=0.76) — consistent
   with the idea that a larger model has more refusal machinery to erode gradually (1.5B tends to refuse or not).

**Caveat.** Absolute projection signs are offset between the single-turn extraction domain and the multi-turn
mhj domain, so the result is the across-turn **trend** (refusal decays, harm flat), not absolute values.
Run 4 (7B, N=120) is kept as an honest `EXTRACTION FAILED` — set-C retention 34<50 made the harmfulness
direction untrustworthy; Run 5 fixes it by raising `N_PROMPTS` to 300.

## Status

- [x] Notebook authored; tensor/hook logic verified; runs top-to-bottom on Kaggle.
- [x] Env fixes baked in: **torch 2.4.1+cu121** (Kaggle CLI P100 has no Pascal kernels in stock torch 2.10);
      AdvBench loaded from the **ungated llm-attacks GitHub CSV** (`walledai/AdvBench` is gated).
- [x] Runs 1–2 (1.5B, seed): clean causal double dissociation; conversation-level attack-success filtering added.
- [x] Run 3 (1.5B, **mhj**): 37/40 attacks succeeded; refusal trajectory now decouples qualitatively.
- [x] Run 4 (7B, mhj, N=120): double dissociation replicates; `EXTRACTION FAILED` on set-C retention 34<50.
- [x] **Run 5 (7B, mhj, N=300): `DECOUPLING OBSERVED`.** Set-C retention 86; refusal decays significantly
      (−2.42/turn, p=0.005) while harmfulness holds (+1.55, p=0.19). **Central hypothesis supported at 7B.**
- [ ] **Next:** robustness — repeat across seeds / more mhj conversations for tighter CIs; probe *why* the
      projection signs are domain-offset (extraction vs measurement); optionally a 14B/larger check.

_Update this section after each run._
