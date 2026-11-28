"""
CG-SAE v2 - shared library.

Rebuild of the minor-project pipeline with the correctness bugs fixed and the missing
controls added. Everything a run needs lives here; the Kaggle kernels are thin drivers.

What changed from v1 (and why)
------------------------------
1. Activations are collected **per token**, not mean-pooled over the sequence. v1 trained
   on mean-pooled sequence vectors and then encoded individual token residuals at steering
   time - different distributions, different scale, and act_mean/act_std came from the
   wrong one.
2. The SAE is **overcomplete** (expansion >= 8) and uses a **TopK** activation, so L0 is
   exact and identical across every arm. v1 had d_latent == d_in == 768 and L0 ~ 394, which
   is a dense autoencoder, not a sparse one. Matched L0 + matched FVU is the only way the
   head-to-head means anything.
3. Steering edits the residual by the **difference** of two decodes,
   h' = h + (dec(z') - dec(z)), so the SAE reconstruction error is preserved and an
   intervention that changes no latent costs exactly zero perplexity. v1 replaced h with a
   raw decode, which charges every intervention the full reconstruction error.
4. Leakage is **scale-invariant** (a ratio of activation mass), so it cannot be driven to
   zero by shrinking the off-group - which is what happened in v1's Exp-1 tau=0.20 config
   (dead B-group, delta = 0.0000).

Author: rebuild for CARE 2026 submission.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    # model / hook
    model_name: str = "gpt2"
    layer: int = 8
    hook: str = "resid_post"

    # concept axis
    axis: str = "sentiment"            # sentiment | toxicity | refusal
    n_docs: int = 4000                 # documents per run, balanced across the two classes
    max_seq_tokens: int = 128          # tokens kept per document
    max_train_tokens: int = 400_000    # cap on the token-activation table

    # SAE
    expansion: int = 8                 # d_latent = expansion * d_in
    topk: int = 32                     # exact L0, identical across every arm
    sae_epochs: int = 12
    sae_batch: int = 4096
    sae_lr: float = 3e-4
    lambda_cg: float = 3.0
    lambda_scar: float = 1.0
    pen_mode: str = "contrast"         # contrast | ratio | mean | sq_mean (v1)
    tag_fracs: tuple = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)

    # evaluation
    n_steer_prompts: int = 120
    gen_tokens: int = 32
    ppl_docs: int = 200
    alphas: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    n_select: int = 64                 # latents intervened on, same count for every SAE arm

    seeds: tuple = (0, 1, 2)
    out_dir: str = "results"
    device: str = "cuda"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ---------------------------------------------------------------------------
# SAE variants
#   All three share encoder/decoder shape and the TopK activation, so L0 is identical
#   by construction and only the training objective differs.
# ---------------------------------------------------------------------------
class TopKSAE(nn.Module):
    """Vanilla TopK sparse autoencoder (Gao et al., arXiv 2406.04093).

    The *unsupervised control*. Paired with post-hoc latent selection it reproduces the
    SAE-SSV recipe (arXiv 2505.16188) - the baseline CG-SAE must beat to claim that
    training-time supervision buys anything.
    """

    kind = "vanilla"

    def __init__(self, d_in: int, d_latent: int, k: int):
        super().__init__()
        self.d_in, self.d_latent, self.k = d_in, d_latent, k
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_in, d_latent))
        self.b_enc = nn.Parameter(torch.zeros(d_latent))
        self.W_dec = nn.Parameter(torch.empty(d_latent, d_in))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self.normalise_decoder()

    @torch.no_grad()
    def normalise_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True) + EPS)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_pre) @ self.W_enc + self.b_enc)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU then exact TopK - every arm has L0 == k."""
        pre = self.encode_pre(x)
        if self.k >= self.d_latent:
            return pre
        vals, idx = torch.topk(pre, self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, vals)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_pre

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def aux_loss(self, z, labels, pre):
        return torch.zeros((), device=z.device), {}


class CGSAE(TopKSAE):
    """Class-Grouped SAE - the method under test.

    The latent space is split into three contiguous blocks:

        G-group  [0, n_tag)                    wanted-class latents
        neutral  [n_tag, d_latent - n_tag)     unpenalised buffer for shared features
        B-group  [d_latent - n_tag, d_latent)  unwanted-class latents

    with n_tag = floor(d_latent * tag_frac). The penalty suppresses a group's activation on
    inputs of the opposing class. tag_frac = 0.5 removes the buffer entirely and is the
    ablation that isolates what the buffer contributes.

    The penalty acts on the **post-TopK** code, i.e. the latents the decoder actually uses.

    Penalty forms (`pen_mode`)
    --------------------------
    `sq_mean`  v1's formulation: (mean off-group activation)^2.
               **Structurally inert under real sparsity.** With L0 = 32 of 6144 latents, a
               group of ~1229 latents is almost entirely zero for any given token, so the
               mean is ~1e-2, the square ~1e-4, and even at lambda = 30 the term is 0.5% of
               the loss. Measured: leakage moves only 0.494 -> 0.435 (chance = 0.5) while
               FVU changes by 1e-4. It worked in v1 only because that model was dense
               (L0 ~ 394 of 768).
    `mean`     unsquared. Gradient does not vanish as the off-group goes quiet.
    `ratio`    the normalised leakage itself - off-group share of tagged activation mass.
               Scale-free, so it cannot be satisfied by shrinking activations, and it does
               not vanish under sparsity because it is a ratio of two equally sparse sums.
    """

    kind = "cgsae"

    def __init__(self, d_in, d_latent, k, tag_frac: float, lambda_cg: float,
                 pen_mode: str = "ratio"):
        super().__init__(d_in, d_latent, k)
        self.tag_frac, self.lambda_cg, self.pen_mode = tag_frac, lambda_cg, pen_mode
        n_tag = int(d_latent * tag_frac)
        self.n_tag = n_tag
        self.register_buffer("g_idx", torch.arange(0, n_tag))
        self.register_buffer("b_idx", torch.arange(d_latent - n_tag, d_latent))
        self.register_buffer("neutral_idx", torch.arange(n_tag, d_latent - n_tag))

    @property
    def buffer_ratio(self) -> float:
        n_neutral = self.d_latent - 2 * self.n_tag
        return n_neutral / (2 * self.n_tag) if self.n_tag else float("inf")

    def aux_loss(self, z, labels, pre):
        pos, neg = labels == 1, labels == 0
        if pos.sum() == 0 or neg.sum() == 0:
            return torch.zeros((), device=z.device), {}

        if self.pen_mode == "contrast":
            # Routing objective. Shares are taken against the *total* activation mass
            # (neutral included), and the two classes are contrasted, so the loss rewards
            # the B-block carrying mass on unwanted inputs as much as it rewards the block
            # being quiet on wanted ones.
            #
            # Every pure suppression penalty ("do not fire off-group") is minimised by
            # silence: kill both tagged blocks, route everything through the neutral
            # buffer, score a perfect leakage of 0 while encoding nothing. That is exactly
            # what v1's Exp-1 tau=0.20 configuration did (dead B-group, delta = 0.0000),
            # and it reappeared here - `ratio` at lambda=1 left 79% of the tagged capacity
            # dead. Contrast cannot be gamed that way: silence scores 0, correct routing
            # scores -1.
            total = z.sum(-1) + EPS
            g_share = z[:, self.g_idx].sum(-1) / total
            b_share = z[:, self.b_idx].sum(-1) / total
            pen = -0.5 * ((b_share[neg].mean() - b_share[pos].mean())
                          + (g_share[pos].mean() - g_share[neg].mean()))
        elif self.pen_mode == "ratio":
            g_mass = z[:, self.g_idx].sum(-1)
            b_mass = z[:, self.b_idx].sum(-1)
            tagged = g_mass + b_mass + EPS
            pen = 0.5 * ((b_mass[pos] / tagged[pos]).mean()
                         + (g_mass[neg] / tagged[neg]).mean())
        else:
            g_on_neg = z[neg][:, self.g_idx].mean()
            b_on_pos = z[pos][:, self.b_idx].mean()
            pen = (g_on_neg ** 2 + b_on_pos ** 2) if self.pen_mode == "sq_mean" \
                else (g_on_neg + b_on_pos)
        return self.lambda_cg * pen, {"cg_pen": float(pen.item())}


class ScarSAE(TopKSAE):
    """Scar-style SAE (arXiv 2411.07122): one latent conditioned on the concept label with
    a BCE term, reimplemented at our scale as a literature baseline.

    The "supervision at minimum granularity" arm. If CG-SAE's blocks-plus-buffer beats it,
    capacity allocation matters; if not, one conditioned latent suffices and CG-SAE adds
    nothing.
    """

    kind = "scar"

    def __init__(self, d_in, d_latent, k, lambda_scar: float):
        super().__init__(d_in, d_latent, k)
        self.lambda_scar = lambda_scar
        self.cond_idx = 0

    def aux_loss(self, z, labels, pre):
        # BCE on the pre-TopK value so the conditioned latent always receives gradient,
        # mirroring the original, where the conditioned unit bypasses the sparsity gate.
        logit = pre[:, self.cond_idx]
        target = (labels == 0).float()      # fires on the unwanted class
        bce = F.binary_cross_entropy_with_logits(logit, target)
        return self.lambda_scar * bce, {"scar_bce": float(bce.item())}


def build_sae(kind: str, d_in: int, d_latent: int, k: int, cfg: RunConfig,
              tag_frac: float | None = None) -> TopKSAE:
    if kind == "vanilla":
        return TopKSAE(d_in, d_latent, k)
    if kind == "cgsae":
        assert tag_frac is not None, "cgsae needs a tag_frac"
        return CGSAE(d_in, d_latent, k, tag_frac, cfg.lambda_cg, cfg.pen_mode)
    if kind == "scar":
        return ScarSAE(d_in, d_latent, k, cfg.lambda_scar)
    raise ValueError(f"unknown SAE kind: {kind}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_sae(sae: TopKSAE, acts: torch.Tensor, labels: torch.Tensor, cfg: RunConfig,
              val_acts: torch.Tensor | None = None, val_labels: torch.Tensor | None = None,
              log_every: int = 2, verbose: bool = True) -> dict:
    """Train one SAE arm. Returns a history dict.

    acts are expected pre-standardised (see standardise_acts) so that FVU is comparable
    across arms and models.
    """
    device = acts.device
    sae.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.sae_lr)
    n = acts.shape[0]
    hist = {"epoch": [], "fvu": [], "aux": [], "val_fvu": [], "val_leak": []}

    for epoch in range(cfg.sae_epochs):
        sae.train()
        perm = torch.randperm(n, device=device)
        tot_fvu, tot_aux, nb = 0.0, 0.0, 0
        for i in range(0, n, cfg.sae_batch):
            idx = perm[i:i + cfg.sae_batch]
            x, y = acts[idx], labels[idx]

            pre = sae.encode_pre(x)
            if sae.k < sae.d_latent:
                vals, ids = torch.topk(pre, sae.k, dim=-1)
                z = torch.zeros_like(pre).scatter_(-1, ids, vals)
            else:
                z = pre
            x_hat = sae.decode(z)

            # FVU rather than raw MSE, so the scale is comparable across arms/models.
            resid = (x - x_hat).pow(2).sum(-1)
            total = (x - x.mean(0, keepdim=True)).pow(2).sum(-1)
            fvu = (resid.mean() / (total.mean() + EPS))

            aux, _ = sae.aux_loss(z, y, pre)
            loss = fvu + aux

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                sae.normalise_decoder()

            tot_fvu += float(fvu.item())
            tot_aux += float(aux.item()) if torch.is_tensor(aux) else float(aux)
            nb += 1

        hist["epoch"].append(epoch)
        hist["fvu"].append(tot_fvu / max(nb, 1))
        hist["aux"].append(tot_aux / max(nb, 1))

        if val_acts is not None:
            m = evaluate_sae(sae, val_acts, val_labels)
            hist["val_fvu"].append(m["fvu"])
            hist["val_leak"].append(m.get("norm_leakage", float("nan")))
        if verbose and (epoch % log_every == 0 or epoch == cfg.sae_epochs - 1):
            tail = ""
            if val_acts is not None:
                tail = f" | val_fvu={hist['val_fvu'][-1]:.4f} leak={hist['val_leak'][-1]:.4f}"
            print(f"    epoch {epoch:3d} | fvu={hist['fvu'][-1]:.4f} aux={hist['aux'][-1]:.4f}{tail}",
                  flush=True)
    return hist


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_sae(sae: TopKSAE, acts: torch.Tensor, labels: torch.Tensor,
                 batch: int = 8192) -> dict:
    """Reconstruction + separation metrics. All scale-invariant."""
    sae.eval()
    n = acts.shape[0]
    resid_sum, total_sum, l0_sum = 0.0, 0.0, 0.0
    z_chunks = []
    for i in range(0, n, batch):
        x = acts[i:i + batch]
        x_hat, z = sae(x)
        resid_sum += float((x - x_hat).pow(2).sum().item())
        total_sum += float((x - acts.mean(0, keepdim=True)).pow(2).sum().item())
        l0_sum += float((z > 0).sum(-1).float().sum().item())
        z_chunks.append(z.cpu())
    z_all = torch.cat(z_chunks)
    out = {
        "fvu": resid_sum / (total_sum + EPS),
        "explained_var": 1.0 - resid_sum / (total_sum + EPS),
        "l0": l0_sum / n,
    }
    if isinstance(sae, CGSAE):
        g_i, b_i = sae.g_idx.cpu(), sae.b_idx.cpu()
        out["norm_leakage"] = normalised_leakage(z_all, labels.cpu(), g_i, b_i)
        out["raw_leakage"] = raw_leakage(z_all, labels.cpu(), g_i, b_i)
        out["tag_frac"] = sae.tag_frac
        out["buffer_ratio"] = sae.buffer_ratio
        out.update(block_routing(z_all, labels.cpu(), g_i, b_i))
    return out


def block_routing(z: torch.Tensor, labels: torch.Tensor,
                  g_idx: torch.Tensor, b_idx: torch.Tensor) -> dict:
    """Is the B-block actually *carrying* the concept, or merely silent?

    Leakage alone cannot answer this. A model that kills both tagged blocks and routes
    everything through the neutral buffer scores a perfect leakage while encoding nothing -
    v1's tau=0.20 failure (dead B-group, delta = 0.0000), which reappeared here with 79-98%
    of tagged capacity dead under suppression-only penalties. Normalised leakage fixes the
    *rescaling* loophole but not this one, because both terms of the ratio vanish together.

    `routing` is the quantity that matters and that silence cannot fake: the extra share of
    activation mass the B-block receives on unwanted inputs relative to wanted ones. Any
    claim about the block must be read next to `b_alive_frac`.
    """
    total = z.sum(-1) + EPS
    b_share = z[:, b_idx].sum(-1) / total
    g_share = z[:, g_idx].sum(-1) / total
    pos, neg = labels == 1, labels == 0
    fires = (z > 0).any(0)
    return {
        "b_share_unwanted": float(b_share[neg].mean()),
        "b_share_wanted": float(b_share[pos].mean()),
        "g_share_wanted": float(g_share[pos].mean()),
        "routing": float(b_share[neg].mean() - b_share[pos].mean()),
        "b_alive_frac": float(fires[b_idx].float().mean()),
        "g_alive_frac": float(fires[g_idx].float().mean()),
        "tagged_mass_frac": float((b_share + g_share).mean()),
    }


def normalised_leakage(z: torch.Tensor, labels: torch.Tensor,
                       g_idx: torch.Tensor, b_idx: torch.Tensor) -> float:
    """Off-group share of tagged activation mass.

    For a wanted input, the leak is the fraction of the tagged activation mass that lands
    in the B-group; for an unwanted input, the fraction landing in the G-group. Being a
    *ratio*, this cannot be driven to zero by shrinking the off-group's activations - the
    failure mode that made v1's raw leakage metric meaningless (a dead B-group scored a
    perfect 0.0 while encoding nothing).

    Random assignment -> ~0.5. Perfect separation -> 0.
    """
    g_mass = z[:, g_idx].sum(-1)
    b_mass = z[:, b_idx].sum(-1)
    tagged = g_mass + b_mass + EPS
    pos, neg = labels == 1, labels == 0
    leaks = []
    if pos.any():
        leaks.append((b_mass[pos] / tagged[pos]).mean().item())
    if neg.any():
        leaks.append((g_mass[neg] / tagged[neg]).mean().item())
    return float(np.mean(leaks)) if leaks else float("nan")


def raw_leakage(z: torch.Tensor, labels: torch.Tensor,
                g_idx: torch.Tensor, b_idx: torch.Tensor) -> float:
    """v1's metric, kept only so the paper can show it is gameable."""
    pos, neg = labels == 1, labels == 0
    vals = []
    if pos.any():
        vals.append(z[pos][:, b_idx].mean().item())
    if neg.any():
        vals.append(z[neg][:, g_idx].mean().item())
    return float(np.mean(vals)) if vals else float("nan")


def auc(scores: np.ndarray, targets: np.ndarray) -> float:
    """Rank-based AUC; targets in {0,1}. Threshold-free, unlike v1's per-class accuracy."""
    from scipy.stats import rankdata
    targets = np.asarray(targets).astype(int)
    n_pos = int((targets == 1).sum())
    n_neg = int((targets == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(np.asarray(scores, dtype=float))
    return float((r[targets == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def encode_to_cpu(sae: TopKSAE, acts: torch.Tensor, chunk: int = 8192,
                  max_rows: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode in chunks, keeping the latent matrix on CPU.

    At d_latent = 12288 a 40k-token latent matrix is ~2 GB, and any elementwise op that
    materialises a second copy on GPU (`(z > 0).float()`, `z[mask]`) doubles it. That is
    what killed the first Qwen run. Encoding chunkwise to CPU costs a PCIe transfer and
    makes the downstream statistics memory-safe at any dictionary size.

    Returns (z_cpu, row_index) so labels can be aligned after subsampling.
    """
    n = acts.shape[0]
    idx = torch.arange(n)
    if max_rows is not None and n > max_rows:
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:max_rows]
        idx, _ = torch.sort(idx)
    out = []
    for i in range(0, len(idx), chunk):
        sel = idx[i:i + chunk].to(acts.device)
        out.append(sae.encode(acts[sel]).cpu())
    return torch.cat(out), idx


@torch.no_grad()
def latent_class_auc(z: torch.Tensor, labels: torch.Tensor,
                     lat_chunk: int = 1024) -> np.ndarray:
    """Per-latent AUC for discriminating unwanted (1) from wanted (0).

    Replaces v1's bad_bias = mu_neg / (mu_pos + mu_neg), which is undefined for dead
    latents (0/0 -> 0 via the epsilon) and rewards silence.

    Ranked in latent-blocks with scipy's vectorised `rankdata(axis=0)`. Tie handling is not
    optional here: under TopK most entries are exactly zero, so zeros are the dominant tie
    group and an argsort-based rank would score them arbitrarily.
    """
    from scipy.stats import rankdata
    z_np = z.cpu().numpy()
    tgt = (labels.cpu().numpy() == 0).astype(int)
    n_pos, n_neg = int((tgt == 1).sum()), int((tgt == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(z_np.shape[1], np.nan)
    pos_mask = tgt == 1
    out = np.empty(z_np.shape[1], dtype=float)
    for s in range(0, z_np.shape[1], lat_chunk):
        block = z_np[:, s:s + lat_chunk]
        r = rankdata(block, axis=0)
        out[s:s + lat_chunk] = (r[pos_mask].sum(0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


@torch.no_grad()
def latent_selectivity(z: torch.Tensor, labels: torch.Tensor,
                       min_freq: float = 1e-3, chunk: int = 8192):
    """Class selectivity of each latent, robust to sparsity.

    Per-token AUC badly understates a sparse feature: a latent firing on 2% of tokens but
    perfectly precise for the unwanted class scores AUC ~ 0.51, because AUC is
    recall-sensitive. This is why the calibration run found *no* latent above AUC 0.60 even
    though the raw residual carries the concept at 0.775 - the metric, not the SAE, was the
    problem.

    Selectivity compares *conditional magnitude* instead:

        sel_j = mean(z_j | unwanted) / (mean(z_j | unwanted) + mean(z_j | wanted))

    which is v1's bad-bias, but gated on a minimum firing rate so that dead and
    near-dead latents (where the ratio is 0/0) cannot be selected. That gate is what v1
    lacked, and is why its tau=0.20 configuration could select a silent B-group.

    Returns (selectivity, firing_rate, eligible_mask).

    Accumulated in row-chunks. The direct form - `z[neg].mean(0)` and
    `(z > 0).float().mean(0)` - each materialise a full-size copy, which is what ran the
    first Qwen run out of GPU memory at d_latent = 12288.
    """
    pos, neg = labels == 1, labels == 0
    d = z.shape[1]
    s_unw = torch.zeros(d, dtype=torch.float64)
    s_wan = torch.zeros(d, dtype=torch.float64)
    nz = torch.zeros(d, dtype=torch.float64)
    n_unw = n_wan = 0
    for i in range(0, z.shape[0], chunk):
        zc = z[i:i + chunk].float()
        pc, nc = pos[i:i + chunk], neg[i:i + chunk]
        if nc.any():
            s_unw += zc[nc].sum(0).double().cpu()
            n_unw += int(nc.sum())
        if pc.any():
            s_wan += zc[pc].sum(0).double().cpu()
            n_wan += int(pc.sum())
        nz += (zc > 0).sum(0).double().cpu()
    m_unw = s_unw / max(n_unw, 1)
    m_wan = s_wan / max(n_wan, 1)
    sel = (m_unw / (m_unw + m_wan + EPS)).numpy()
    freq = (nz / z.shape[0]).numpy()
    return sel, freq, freq >= min_freq


def select_latents(z: torch.Tensor, labels: torch.Tensor, n_select: int,
                   restrict: torch.Tensor | None = None,
                   min_freq: float = 1e-3) -> np.ndarray:
    """Pick the n_select latents most selective for the unwanted class.

    `restrict` limits the pool to a group (CG-SAE's B-block); with restrict=None this is
    the post-hoc selection every arm gets, so the selection *method* is held constant and
    the only difference between arms is how the SAE was trained.

    Latents firing on fewer than `min_freq` of tokens are excluded before ranking; if too
    few survive the floor is dropped so a selection is always returned.
    """
    sel, freq, eligible = latent_selectivity(z, labels, min_freq)
    pool = np.arange(z.shape[1]) if restrict is None else restrict.cpu().numpy()
    live = pool[eligible[pool]]
    if len(live) < n_select:                      # floor too strict for this pool
        live = pool[np.argsort(freq[pool])[::-1][:max(n_select * 4, n_select)]]
    ranked = live[np.argsort(sel[live])[::-1]]
    return ranked[:n_select]
