"""
CG-SAE v2 - model-side utilities: activation capture, interventions, evaluation.

Deliberately built on plain HuggingFace transformers + forward hooks rather than
TransformerLens, so the same code path runs on GPT-2 and on Qwen2.5 without a model
conversion step.

The central design decision is in `sae_edit`: an intervention is applied as a **delta
between two decodes**,

    h' = h + (dec(z') - dec(z)) * sigma

so the SAE reconstruction error never enters the residual stream. An intervention that
changes no latent is then exactly a no-op, and the measured perplexity cost is the cost of
*the intervention* rather than the cost of the autoencoder. v1 replaced the residual with a
raw decode and so paid the full reconstruction error on every edit - which is most of what
its reported perplexity increase was measuring.
"""

from __future__ import annotations

import contextlib
import math
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class HookedLM:
    """Thin wrapper exposing one residual-stream hook point per layer."""

    def __init__(self, model_name: str, device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            self.blocks = self.model.transformer.h              # GPT-2 family
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.blocks = self.model.model.layers               # Llama / Qwen family
        else:
            raise ValueError(f"unrecognised architecture for {model_name}")

        self.n_layers = len(self.blocks)
        self.d_model = self.model.config.hidden_size

    # -- capture ------------------------------------------------------------
    @torch.no_grad()
    def capture(self, input_ids: torch.Tensor, layer: int,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Residual stream after `layer`. Returns [batch, seq, d_model]."""
        store = {}

        def hook(_mod, _inp, out):
            store["h"] = (out[0] if isinstance(out, tuple) else out).detach()

        handle = self.blocks[layer].register_forward_hook(hook)
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()
        return store["h"]

    # -- intervene ----------------------------------------------------------
    @contextlib.contextmanager
    def edit(self, layer: int, edit_fn: Callable[[torch.Tensor], torch.Tensor] | None):
        """Context manager applying `edit_fn` to the residual stream after `layer`."""
        if edit_fn is None:
            yield
            return

        def hook(_mod, _inp, out):
            if isinstance(out, tuple):
                return (edit_fn(out[0]),) + tuple(out[1:])
            return edit_fn(out)

        handle = self.blocks[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    # -- generation ---------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompts: Sequence[str], layer: int,
                 edit_fn: Callable | None = None, max_new_tokens: int = 32,
                 batch_size: int = 16) -> list[str]:
        """Greedy decoding, identical for every arm.

        v1 sampled the unpatched completions (temperature 0.7, top_p 0.9) but took argmax
        for the patched ones, so its shift counts compared two different decoders. Greedy
        everywhere removes that confound and makes runs deterministic.
        """
        self.tok.padding_side = "left"
        outs: list[str] = []
        for i in range(0, len(prompts), batch_size):
            chunk = list(prompts[i:i + batch_size])
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to(self.device)
            with self.edit(layer, edit_fn):
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    num_beams=1, pad_token_id=self.tok.pad_token_id)
            for j in range(len(chunk)):
                new = gen[j, enc["input_ids"].shape[1]:]
                outs.append(self.tok.decode(new, skip_special_tokens=True).strip())
        return outs

    # -- perplexity ---------------------------------------------------------
    @torch.no_grad()
    def perplexity(self, texts: Sequence[str], layer: int,
                   edit_fn: Callable | None = None, max_length: int = 128,
                   batch_size: int = 8) -> float:
        """Token-level perplexity on held-out text, with the edit applied at *every*
        position.

        v1 measured perplexity on the ten positive prompts themselves, patching only the
        last five positions - so most tokens were bit-identical by causality and the
        number understated the damage. Here the corpus is held out and the whole sequence
        is edited.
        """
        self.tok.padding_side = "right"
        total_nll, total_tok = 0.0, 0
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(self.device)
            ids, mask = enc["input_ids"], enc["attention_mask"]
            if ids.shape[1] < 2:
                continue
            with self.edit(layer, edit_fn):
                logits = self.model(input_ids=ids, attention_mask=mask).logits
            lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
            tgt = ids[:, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            m = mask[:, 1:].float()
            total_nll += float(-(tok_lp * m).sum().item())
            total_tok += int(m.sum().item())
        return float(math.exp(total_nll / max(total_tok, 1)))


# ---------------------------------------------------------------------------
# Activation table (per token)
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_token_activations(lm: HookedLM, texts: Sequence[str], labels: Sequence[int],
                              layer: int, max_seq_tokens: int = 128,
                              batch_size: int = 16, max_tokens: int | None = None,
                              skip_bos: bool = True):
    """Build a [N_tokens, d_model] table of residual activations with per-token labels.

    Each token inherits its document's class label, as in Scar (arXiv 2411.07122). Padding
    is masked out; the first position is dropped by default since it carries the well-known
    attention-sink outlier that dominates SAE reconstruction error.
    """
    lm.tok.padding_side = "right"
    acts_all, lab_all, doc_all = [], [], []
    n_kept = 0
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i:i + batch_size])
        chunk_lab = list(labels[i:i + batch_size])
        enc = lm.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                     max_length=max_seq_tokens).to(lm.device)
        h = lm.capture(enc["input_ids"], layer, enc["attention_mask"])
        mask = enc["attention_mask"].bool()
        if skip_bos:
            mask[:, 0] = False
        for j in range(len(chunk)):
            sel = h[j][mask[j]]
            if sel.numel() == 0:
                continue
            acts_all.append(sel.float().cpu())
            lab_all.append(torch.full((sel.shape[0],), int(chunk_lab[j]), dtype=torch.long))
            doc_all.append(torch.full((sel.shape[0],), i + j, dtype=torch.long))
            n_kept += sel.shape[0]
        if max_tokens is not None and n_kept >= max_tokens:
            break
    acts = torch.cat(acts_all)
    labs = torch.cat(lab_all)
    docs = torch.cat(doc_all)
    if max_tokens is not None and acts.shape[0] > max_tokens:
        idx = torch.randperm(acts.shape[0])[:max_tokens]
        acts, labs, docs = acts[idx], labs[idx], docs[idx]
    return acts, labs, docs


def standardise(acts: torch.Tensor):
    """Centre and scale to unit mean-norm.

    Per-dimension z-scoring (v1's choice) distorts the residual geometry that steering
    relies on; scaling by a single scalar preserves directions.
    """
    mu = acts.mean(0, keepdim=True)
    centred = acts - mu
    sigma = centred.norm(dim=-1).mean() / math.sqrt(acts.shape[1])
    return (centred / (sigma + EPS)), mu, sigma


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------
def make_sae_edit(sae, latent_idx, alpha: float, mu: torch.Tensor, sigma: torch.Tensor):
    """Suppress `latent_idx` by factor alpha (0 = no-op, 1 = fully zeroed).

    Applied as a delta between decodes so the reconstruction error is preserved and
    alpha = 0 is exactly the identity.
    """
    if alpha == 0.0:
        return None
    idx = torch.as_tensor(np.asarray(latent_idx, dtype=np.int64), device=mu.device)

    def edit(h: torch.Tensor) -> torch.Tensor:
        dt = h.dtype
        x = (h.float() - mu) / sigma
        z = sae.encode(x)
        z2 = z.clone()
        z2[..., idx] = z2[..., idx] * (1.0 - alpha)
        delta = (sae.decode(z2) - sae.decode(z)) * sigma
        return (h.float() + delta).to(dt)

    return edit


def make_direction_edit(direction: torch.Tensor, alpha: float, scale: float):
    """Difference-in-means / CAA steering: h' = h - alpha * scale * v_hat.

    The standard non-SAE baseline (Turner et al. 2308.10248; Arditi et al. NeurIPS 2024).
    arXiv 2510.01246 finds this beats SAE steering on several task families, so it is a
    real competitor rather than a formality.
    """
    if alpha == 0.0:
        return None
    v = direction / (direction.norm() + EPS)

    def edit(h: torch.Tensor) -> torch.Tensor:
        dt = h.dtype
        return (h.float() - alpha * scale * v).to(dt)

    return edit


def make_ablation_edit(direction: torch.Tensor, alpha: float):
    """Directional ablation: project the component along `direction` out of the residual."""
    if alpha == 0.0:
        return None
    v = direction / (direction.norm() + EPS)

    def edit(h: torch.Tensor) -> torch.Tensor:
        dt = h.dtype
        hf = h.float()
        proj = (hf @ v).unsqueeze(-1) * v
        return (hf - alpha * proj).to(dt)

    return edit


def diff_in_means_direction(acts: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """mean(unwanted) - mean(wanted), the classic concept direction."""
    return acts[labels == 0].mean(0) - acts[labels == 1].mean(0)


def probe_direction(acts: torch.Tensor, labels: torch.Tensor, max_n: int = 60000):
    """Logistic-regression weight vector - a stronger linear baseline than diff-in-means."""
    from sklearn.linear_model import LogisticRegression
    x = acts.cpu().numpy()
    y = (labels.cpu().numpy() == 0).astype(int)
    if len(x) > max_n:
        sel = np.random.RandomState(0).choice(len(x), max_n, replace=False)
        x, y = x[sel], y[sel]
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(x, y)
    return torch.tensor(lr.coef_[0], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------
class Judge:
    """External classifier scoring generated text on the concept axis.

    v1 counted hits from a 35-word hand-written vocabulary list, which cannot distinguish
    "not good" from "good" and shares tokens with the prompts it scores.
    """

    MODELS = {
        "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
        "toxicity": "s-nlp/roberta_toxicity_classifier",
    }

    def __init__(self, axis: str, device: str = "cuda", batch_size: int = 32):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = self.MODELS[axis]
        self.axis = axis
        self.device = device
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.unwanted_id = self._unwanted_label_id()

    def _unwanted_label_id(self) -> int:
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        for i, lab in id2label.items():
            if lab in ("negative", "toxic", "label_1") and self.axis == "sentiment" and lab == "negative":
                return i
            if self.axis == "toxicity" and lab in ("toxic", "label_1"):
                return i
        return 0 if self.axis == "sentiment" else 1

    @torch.no_grad()
    def score(self, texts: Sequence[str]) -> np.ndarray:
        """P(unwanted class) per text, in [0, 1]."""
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [t if t.strip() else "." for t in texts[i:i + self.batch_size]]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to(self.device)
            probs = self.model(**enc).logits.softmax(-1)
            out.append(probs[:, self.unwanted_id].float().cpu().numpy())
        return np.concatenate(out) if out else np.array([])
