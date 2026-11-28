import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.44.2','datasets==2.20.0'],check=False)
"""
Buckets inside the language model itself, then physical pruning.

This tests the original CG idea directly (no SAE): partition MLP neurons of a transformer
into G / neutral / B blocks, train the model with the LM loss plus the contrastive routing
penalty on those neurons, then DELETE the B-block (zero its in/out weights) and check:

  1. is the unwanted behaviour gone?          (judge score on negative-review continuations)
  2. can the model still write?               (WikiText perplexity)
  3. was only the unwanted knowledge removed? (perplexity on held-out negative vs positive reviews)
  4. is it really gone, or hidden?            (relearning: brief retraining on negative text)

Two modes:
  --mode finetune : GPT-2 small, buckets in MLP layers 5-9, fine-tuned on IMDB
  --mode scratch  : small GPT trained from scratch on IMDB, buckets in MLP layers 1-4

Conditions per seed:
  plain  : same training, lambda = 0                (partition without the penalty)
  bucket : lambda > 0
  and prunes: none / B-block / G-block / random block of equal width.
"""


import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

EPS = 1e-8
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_data(n_train, n_prompts, seed=0):
    from datasets import load_dataset
    tr = load_dataset("imdb", split="train")
    te = load_dataset("imdb", split="test")
    rng = random.Random(seed)
    clean = lambda t: t.replace("<br />", " ")
    pos = [clean(r["text"]) for r in tr if r["label"] == 1]
    neg = [clean(r["text"]) for r in tr if r["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    k = n_train // 2
    train = [(t, 1) for t in pos[:k]] + [(t, 0) for t in neg[:k]]
    rng.shuffle(train)
    tpos = [clean(r["text"]) for r in te if r["label"] == 1]
    tneg = [clean(r["text"]) for r in te if r["label"] == 0]
    rng.shuffle(tpos); rng.shuffle(tneg)

    def openings(xs, n):
        out = []
        for t in xs:
            w = t.split()
            if len(w) >= 14:
                out.append(" ".join(w[:10]))
            if len(out) >= n:
                break
        return out
    return dict(train=train,
                p_neg=openings(tneg, n_prompts), p_pos=openings(tpos, n_prompts),
                ev_neg=tneg[-300:], ev_pos=tpos[-300:],
                # Held-out test negatives, disjoint from the prompts (taken from the front)
                # and from ev_neg (the last 300). Taking them from the train split after the
                # first k reviews left nothing when n_train used the whole split, and the
                # relearning loop then waited forever for a batch.
                relearn=tneg[1000:3000])


def load_wiki(n=150):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    out = [r["text"].strip() for r in ds
           if len(r["text"].split()) >= 40 and not r["text"].strip().startswith("=")]
    random.Random(0).shuffle(out)
    return out[:n]


# ---------------------------------------------------------------------------
# model + buckets
# ---------------------------------------------------------------------------
def make_model(mode, tok, n_embd=256, n_layer=6):
    from transformers import GPT2Config, GPT2LMHeadModel
    if mode == "finetune":
        return GPT2LMHeadModel.from_pretrained("gpt2")
    cfg = GPT2Config(vocab_size=len(tok), n_positions=256, n_embd=n_embd, n_layer=n_layer, n_head=8,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    return GPT2LMHeadModel(cfg)


class Buckets:
    """Contiguous G / neutral / B partition of the MLP hidden units in chosen layers."""

    def __init__(self, model, layers, tau):
        self.layers = layers
        h = model.config.n_inner or 4 * model.config.n_embd
        self.h = h
        n = int(h * tau)
        self.n = n
        self.g = torch.arange(0, n)
        self.b = torch.arange(h - n, h)
        self.acts = {}
        self.handles = []
        for L in layers:
            mod = model.transformer.h[L].mlp.act
            self.handles.append(mod.register_forward_hook(self._hook(L)))

    def _hook(self, L):
        def f(_m, _i, out):
            self.acts[L] = out
        return f

    def penalty(self, labels, mask):
        """Contrastive routing penalty, averaged over bucketed layers. Silence scores 0."""
        pens = []
        y = labels[:, None].expand_as(mask)[mask]
        pos, neg = y == 1, y == 0
        if pos.sum() == 0 or neg.sum() == 0:
            return torch.zeros((), device=mask.device)
        for L in self.layers:
            a = self.acts[L][mask].clamp(min=0)                       # [tokens, h]
            tot = a.sum(-1) + EPS
            gs = a[:, self.g.to(a.device)].sum(-1) / tot
            bs = a[:, self.b.to(a.device)].sum(-1) / tot
            pens.append(-0.5 * ((bs[neg].mean() - bs[pos].mean())
                                + (gs[pos].mean() - gs[neg].mean())))
        return torch.stack(pens).mean()

    @torch.no_grad()
    def routing(self, model, tok, texts_pos, texts_neg):
        vals = []
        for texts, lab in [(texts_pos, 1), (texts_neg, 0)]:
            enc = tok(texts[:100], return_tensors="pt", padding=True, truncation=True,
                      max_length=128).to(DEV)
            model(**enc)
            m = enc["attention_mask"].bool()
            per = []
            for L in self.layers:
                a = self.acts[L][m].clamp(min=0)
                per.append((a[:, self.b.to(a.device)].sum(-1) / (a.sum(-1) + EPS)).mean().item())
            vals.append(np.mean(per))
        return vals[1] - vals[0]                                        # B share: unwanted - wanted

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def prune(model, layers, idx):
    """Physically delete MLP neurons: zero their input weights, bias and output weights."""
    m = copy.deepcopy(model)
    idx = torch.as_tensor(idx)
    for L in layers:
        mlp = m.transformer.h[L].mlp
        idx = idx.to(mlp.c_fc.weight.device)
        mlp.c_fc.weight[:, idx] = 0          # Conv1D: [in, out]
        mlp.c_fc.bias[idx] = 0
        mlp.c_proj.weight[idx, :] = 0        # Conv1D: [in, out]
    return m


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(model, tok, data, lam, layers, tau, steps, lr, bs, seed, max_len=128):
    set_seed(seed)
    model.to(DEV).train()
    bk = Buckets(model, layers, tau)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(0.05, 1 - s / steps))
    items = list(data)
    if len(items) < bs:
        raise ValueError(f"train() got {len(items)} examples for batch size {bs}; "
                         "refusing to loop forever on an empty epoch")
    step, t0 = 0, time.time()
    while step < steps:
        random.shuffle(items)
        for i in range(0, len(items) - bs + 1, bs):
            batch = items[i:i + bs]
            enc = tok([t for t, _ in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(DEV)
            labels = torch.tensor([y for _, y in batch], device=DEV)
            ids, mask = enc["input_ids"], enc["attention_mask"]
            tgt = ids.masked_fill(mask == 0, -100)
            out = model(input_ids=ids, attention_mask=mask, labels=tgt)
            loss = out.loss
            pen = bk.penalty(labels, mask.bool()) if lam > 0 else torch.zeros((), device=DEV)
            (loss + lam * pen).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 200 == 0:
                log(f"   step {step}/{steps} lm={loss.item():.3f} pen={pen.item():+.3f} "
                    f"({time.time() - t0:.0f}s)")
            step += 1
            if step >= steps:
                break
    model.eval()
    return model, bk


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
class Judge:
    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        n = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tok = AutoTokenizer.from_pretrained(n)
        self.m = AutoModelForSequenceClassification.from_pretrained(n).to(DEV).eval()

    @torch.no_grad()
    def neg(self, texts):
        out = []
        for i in range(0, len(texts), 32):
            c = [t if t.strip() else "." for t in texts[i:i + 32]]
            e = self.tok(c, return_tensors="pt", padding=True, truncation=True,
                         max_length=256).to(DEV)
            out.append(self.m(**e).logits.softmax(-1)[:, 0].cpu().numpy())   # 0 = NEGATIVE
        return float(np.concatenate(out).mean())


@torch.no_grad()
def generate(model, tok, prompts, n_new=32):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), 24):
        c = prompts[i:i + 24]
        e = tok(c, return_tensors="pt", padding=True).to(DEV)
        g = model.generate(**e, max_new_tokens=n_new, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        for j in range(len(c)):
            outs.append(tok.decode(g[j, e["input_ids"].shape[1]:], skip_special_tokens=True))
    tok.padding_side = "right"
    return outs


@torch.no_grad()
def ppl(model, tok, texts, max_len=128):
    tok.padding_side = "right"
    nll, n = 0.0, 0
    for i in range(0, len(texts), 16):
        e = tok(texts[i:i + 16], return_tensors="pt", padding=True, truncation=True,
                max_length=max_len).to(DEV)
        lg = model(**e).logits[:, :-1].float()
        t = e["input_ids"][:, 1:]
        m = e["attention_mask"][:, 1:].float()
        lp = F.log_softmax(lg, -1).gather(-1, t[..., None]).squeeze(-1)
        nll += float(-(lp * m).sum()); n += int(m.sum())
    return math.exp(nll / max(n, 1))


def evaluate(model, tok, judge, d, wiki, keep_gen=False):
    g_neg = generate(model, tok, d["p_neg"])
    g_pos = generate(model, tok, d["p_pos"])
    r = dict(neg_on_negprompts=judge.neg(g_neg), neg_on_posprompts=judge.neg(g_pos),
             ppl_wiki=ppl(model, tok, wiki), ppl_imdb_neg=ppl(model, tok, d["ev_neg"]),
             ppl_imdb_pos=ppl(model, tok, d["ev_pos"]))
    if keep_gen:
        r["examples"] = list(zip(d["p_neg"][:4], g_neg[:4]))
    return r


def relearn(model, tok, judge, d, steps, lr, bs, seed):
    """Brief retraining on negative reviews: how fast does the removed behaviour return?"""
    m = copy.deepcopy(model)
    m, bk = train(m, tok, [(t, 0) for t in d["relearn"]], 0.0, [], 0.1, steps, lr, bs, seed)
    bk.remove()
    return judge.neg(generate(m, tok, d["p_neg"]))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="finetune", choices=["finetune", "scratch"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--lams", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-prompts", type=int, default=120)
    ap.add_argument("--relearn-steps", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--check", action="store_true",
                    help="exercise every stage with the REAL data settings but only a few "
                         "steps. The earlier smoke test shrank the data and so missed the "
                         "empty-relearning-set bug that only appears at full size.")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly before starting a seed once this much time has "
                         "passed, so results are saved instead of lost to a timeout")
    ap.add_argument("--out", default="/kaggle/working/results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    log("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no GPU!)")
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU visible - refusing to run on CPU")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    if a.mode == "finetune":
        layers, steps, lr, bs = [5, 6, 7, 8, 9], 600, 5e-5, 16
    else:
        layers, steps, lr, bs = [1, 2, 3, 4], 4000, 6e-4, 32
    if a.layers: layers = a.layers
    if a.steps: steps = a.steps
    if a.smoke:
        a.n_train, a.n_prompts, a.seeds, steps, a.relearn_steps = 400, 16, [0], 30, 10
    if a.check:
        a.n_prompts, a.seeds, steps, a.relearn_steps = 8, a.seeds[:1], 5, 3

    d = load_data(a.n_train, a.n_prompts)
    wiki = load_wiki(10 if (a.smoke or a.check) else 150)
    sizes = {k: len(v) for k, v in d.items()}
    sizes["wiki"] = len(wiki)
    log("data sizes:", sizes)
    for k, v in sizes.items():
        if v < bs and not ((k.startswith("p_") or k == "wiki") and (a.smoke or a.check)):
            raise ValueError(f"data split '{k}' has only {v} items (< batch {bs}) - aborting")
    judge = Judge()
    log(f"mode={a.mode} layers={layers} tau={a.tau} lams={a.lams} steps={steps} "
        f"train_docs={len(d['train'])} max_hours={a.max_hours}")

    rows, examples = [], {}
    for seed in a.seeds:
        if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
            log(f"time limit {a.max_hours} h reached - stopping before seed {seed}; "
                f"results so far are saved")
            break
        log(f"===== seed {seed} =====")
        base = make_model(a.mode, tok, a.n_embd, a.n_layer)
        H = (base.config.n_inner or 4 * base.config.n_embd)
        if a.mode == "finetune":
            r = evaluate(base.to(DEV).eval(), tok, judge, d, wiki)
            rows.append(dict(seed=seed, model="pretrained", prune="none", **r))
            log("   pretrained:", {k: round(v, 3) for k, v in r.items()})

        models = {}
        for name, lam in [("plain", 0.0)] + [(f"bucket_l{l:g}", l) for l in a.lams]:
            log(f" training {name} (lambda={lam})")
            m, bk = train(make_model(a.mode, tok, a.n_embd, a.n_layer), tok, d["train"], lam, layers, a.tau,
                          steps, lr, bs, seed)
            rout = bk.routing(m, tok, [t for t, y in d["train"][:200] if y == 1],
                              [t for t, y in d["train"][:200] if y == 0])
            bk.remove()
            models[name] = (m, bk, rout)
            log(f"   {name} routing={rout:+.3f}")

        rng = np.random.RandomState(seed + 99)
        n_tag = int(H * a.tau)
        rand_idx = rng.choice(H, n_tag, replace=False)
        for name, (m, bk, rout) in models.items():
            conds = [("none", None), ("B", bk.b), ("G", bk.g), ("random", rand_idx)]
            if name == "plain":
                conds = [("none", None), ("B", bk.b), ("random", rand_idx)]
            for cname, idx in conds:
                mm = m if idx is None else prune(m, layers, idx).to(DEV).eval()
                keep = (name.startswith("bucket") and seed == a.seeds[0])
                r = evaluate(mm, tok, judge, d, wiki, keep_gen=keep)
                if keep:
                    examples[f"{name}/{cname}"] = r.pop("examples")
                if cname in ("B", "random") and a.relearn_steps > 0:
                    r["neg_after_relearn"] = relearn(mm, tok, judge, d, a.relearn_steps,
                                                     lr, bs, seed)
                rows.append(dict(seed=seed, model=name, lam=float(name.split("_l")[1]) if "_l" in name else 0.0, prune=cname, routing=rout,
                                 n_pruned_per_layer=0 if idx is None else len(idx), **r))
                log(f"   {name}/{cname}:", {k: round(v, 3) for k, v in r.items()
                                              if isinstance(v, float)})
                if idx is not None:
                    del mm
                    torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(f"{a.out}/bucket_llm_{a.mode}{a.tag}.csv", index=False)
        del models
        torch.cuda.empty_cache()

    json.dump(examples, open(f"{a.out}/bucket_llm_{a.mode}{a.tag}_examples.json", "w"), indent=1)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/bucket_llm_{a.mode}{a.tag}.csv", index=False)
    cols = ["neg_on_negprompts", "neg_on_posprompts", "ppl_wiki", "ppl_imdb_neg",
            "ppl_imdb_pos", "neg_after_relearn"]
    print("\n==== SUMMARY (mean over seeds) ====")
    print(df.groupby(["model", "prune"])[[c for c in cols if c in df]].mean()
          .round(3).to_string())
    log("done")




for argv in [["--mode", "finetune", "--steps", "1500", "--n-train", "25000", "--lams", "3", "--tag", "_check", "--check"], ["--mode", "scratch", "--n-embd", "512", "--n-layer", "8", "--layers", "2", "3", "4", "5", "--n-train", "25000", "--steps", "4000", "--lams", "1", "--tag", "_checks", "--check"]]:
    sys.argv = ['bucket_llm.py'] + argv
    print('ARGS', argv, flush=True)
    main()
