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
| **DECOUPLING OBSERVED** | refusal falls, harmfulness holds → hypothesis supported, project is live |
| **BOTH DECAY** | both fall → hypothesis dead |
| **NO SIGNAL** | neither moves → nothing happening in these representations |
| **EXTRACTION FAILED** | directions are the same thing, or set-C jailbreaks didn't work → measurement broken, no conclusion |

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

---

## Repository layout
```
major_project/
├── README.md                     # this file
├── notebooks/
│   └── exp1_decoupling.ipynb      # the deliverable — runs top-to-bottom on Kaggle
├── data/
│   └── conversations.json         # 20 attack (crescendo) + 20 benign, matched seed set
├── results/                       # paste Kaggle outputs here (projections.csv, verdict.txt, ...)
├── figures/                       # paste plots here (fig1..fig4 .png)
└── notes/
    └── experiment_log.md          # record each run
```

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
| `Qwen/Qwen2.5-1.5B-Instruct` (default) | fp16 | 1× T4 | **~20–35 min** |
| `Qwen/Qwen2.5-7B-Instruct` | 8-bit (`USE_8BIT=True`) | shards over 2× T4 | **~2–3.5 hours** |

Multi-turn generation (Section 8) dominates both. With 30 GPU-hours/week, the 1.5B pilot costs well
under an hour and the 7B run is comfortably affordable. If the 7B config OOMs in fp16 the loader falls
back to 8-bit automatically and prints a note.

---

## Config knobs (Section 0 of the notebook)
- `MODEL_NAME` — default `Qwen/Qwen2.5-1.5B-Instruct`; 7B commented beside it. **Do not use Llama** — it
  is gated on HuggingFace and approval takes days.
- `USE_8BIT` — 8-bit load for the 7B config.
- `N_PROMPTS` (200), `SEED` (42, seeds torch/numpy/random; greedy decoding = deterministic).
- Toggles: `RUN_BEHAVIORAL_VERIFICATION`, `GENERATE_ASSISTANT_TURNS`, `RELOAD_CACHED_DIRECTIONS`,
  `ABLATE_ALL_LAYERS`. On a fresh session leave them at defaults.
- Guards: `MIN_SETC_RETENTION` (50), `COSINE_SAME_THRESH` (0.9).

---

## Status

- [x] Notebook authored, all code cells compile, tensor/hook logic statically verified.
- [x] Seed conversations authored and matched (turn counts equal, token delta ≤ 20%).
- [x] **First Kaggle run complete — 2026-08-26 (P100 via CLI).** Two environment fixes were needed and
      are now baked into the notebook: (1) Kaggle's CLI always assigns a **P100 (sm_60)** and its stock
      torch 2.10 has no Pascal kernels, so the notebook pins **torch 2.4.1+cu121**; (2) `walledai/AdvBench`
      became **gated**, so AdvBench now loads from the ungated llm-attacks GitHub CSV.
- [x] Directions separate (cosine@layer24 = **−0.315**) and behaviourally verified (retention A/B/C =
      199/187/56). **Causal validation passed with a clean double dissociation** — ablating refusal drops
      refusal behaviour 1.00→0.20 while harmfulness persists; ablating harmfulness zeroes harm-proj while
      refusal is untouched. The base-paper claim (two separable directions) is **causally supported**.
- [x] **Run 2 (2026-08-27): added conversation-level attack-success filtering.** The notebook now
      generates the model's reply each turn, classifies refusal, and splits conversations into
      attack_success / attack_fail / benign; the verdict is computed on *successful* attacks only.
      **8/20 attacks succeeded.** At the harmful turn, successful jailbreaks show refusal ≈**30** vs failed
      ≈**57** (roughly halved, success-vs-fail p<0.001) while harmfulness stays positive — a
      conversation-level replication of the base paper's "jailbreaks reduce refusal without reversing the
      harmfulness belief." Verdict is **NO SIGNAL for across-turn decay**: the crescendo design still
      concentrates the harmful ask in the final turn, so there is no sustained-harmful trajectory to erode.
- [ ] **Next: MultiTurnPSB + 7B.** Switch attacks to the released MultiTurnPSB benchmark (464 validated
      4-turn harmful conversations that escalate gradually), and re-run at Qwen2.5-7B-Instruct (STAR saw
      decay on 8B+; 1.5B may lack the refusal machinery to erode gradually). Same pipeline + attack-success gate.

_Update this section after each run._
