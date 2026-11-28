# Literature positioning — where CG-SAE actually sits

Checked 2026-09-16. The question: is "supervise the SAE latent space with class labels"
already published? **Yes — several times.** The contribution has to be narrowed to the part
that is genuinely unclaimed.

## Prior work that does supervised / concept-structured SAEs

| Work | Where supervision enters | Latent structure imposed | Notes |
|---|---|---|---|
| **Scar** — Sparse Conditioned Autoencoders, arXiv **2411.07122** | during training | **one** latent `h0`, BCE against the token's concept label | Llama3-8B L25 FF, d_latent 24576, TopK=2048. Toxicity / safety / Shakespeare. Steering: ~15% avg toxicity reduction (30% on highly toxic), vs ~5% for prior methods. Detection: up to 98% fewer nodes than an unconditioned SAE. **This is the nearest neighbour.** |
| **GSAE** — Graph-Regularized SAEs for LLM Safety Steering, arXiv **2512.06655** | during training | probe cross-entropy on safety label + graph-Laplacian term | Llama-2 / Mistral-7B / Qwen / Phi; JailbreakBench, HarmBench, TruthfulQA, GSM8K. |
| **SAEParate** — Disentangled Sparse Representations, arXiv **2605.12122** | during training | concept-aware **contrastive** loss on normalised latents | diffusion unlearning; pulls same-label latents together, pushes different-label apart. |
| **SAE-SSV** — Supervised Steering in Sparse Representation Spaces, arXiv **2505.16188** | **after** training | post-hoc supervised *selection* of latents from a pretrained SAE | Gemma / LLaMA; refusal, bias, behaviour tasks. |
| **SSAE** — Specialized SAEs for rare concepts, NAACL Findings **2025** | separate SAE per subdomain | — | tail-concept coverage, not polarity separation. |
| **Group-SAE**, arXiv **2410.21508** | — | groups **layers**, not concepts | name collision only; unrelated idea. |

## What this rules out

The minor-project report's framing — "SAEs offer no guarantee features are organised along
semantic boundaries, so we add class supervision" — is **no longer a novel observation**.
Scar makes exactly that argument and implements it. A reviewer who knows this literature
would reject the current framing as unaware of prior art.

## What is left, and is actually unclaimed

Scar conditions **one** latent on the concept. SAE-SSV selects latents after the fact.
Neither asks the **capacity-allocation** question:

> How many latents should a concept get, and where do features shared between the two
> opposing classes live?

CG-SAE's tag fraction τ allocates ⌊τ·m⌋ latents to each class and leaves the remaining
m − 2⌊τ·m⌋ as an **unpenalised neutral buffer**. That buffer, and the trade-off it governs
(class separation vs reconstruction fidelity), is the part no prior work has.

**Revised claim to test:** *supervision granularity matters* — a tagged block plus a neutral
buffer should beat both (a) one conditioned latent (Scar) and (b) post-hoc selection on a
vanilla SAE (SAE-SSV), at matched sparsity and matched reconstruction. And the buffer ratio,
not the penalty weight, should be the knob that controls it.

This is falsifiable, and the honest negative outcome ("allocation doesn't matter; post-hoc
selection on a vanilla SAE does just as well") is itself a useful result, because three of
the papers above assume the opposite without testing it.

## Baselines this forces into the design

1. **Vanilla TopK SAE + post-hoc top-k selection** — the SAE-SSV control. Mandatory.
2. **Scar-style single conditioned latent** — reimplemented at our scale.
3. **Difference-in-means / CAA steering** on the raw residual (Turner et al. 2308.10248;
   Arditi et al. NeurIPS 2024). Note arXiv **2510.01246** finds mean-activation-difference
   *beats* SAE steering on several task families — so this is a strong baseline, not a formality.
4. **Random-k latents** — null control.
5. All of the above at **matched L0 and matched FVU**, or the comparison means nothing.

## Citation errors to fix in the report

- Ref [13] "Templeton, A., et al.: Scaling and Evaluating Sparse Autoencoders, arXiv:2406.04093"
  is **Gao et al.** (OpenAI), not Templeton. Templeton et al. 2024 is *Scaling Monosemanticity*.
