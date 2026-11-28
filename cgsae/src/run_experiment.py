"""
CG-SAE v2 - main experiment driver.

Runs the head-to-head that the minor project was missing:

    vanilla TopK SAE + post-hoc selection   (= the SAE-SSV recipe, arXiv 2505.16188)
    Scar-style single conditioned latent    (arXiv 2411.07122)
    CG-SAE, swept over tag fraction          (the method under test)
    difference-in-means / CAA steering       (Turner et al.; Arditi et al.)
    logistic-probe direction
    random latents / random direction        (null controls)

all at **matched L0** (TopK, same k) and with reconstruction (FVU) reported so that any
separation advantage can be read against what it cost.

The headline output is a trade-off curve: steering effect (external judge) against
perplexity on off-axis held-out text, swept over intervention strength. A method that
merely intervenes less cannot win on that plot, which is the flaw in v1's single-point
"+31.6% perplexity" comparison.

Usage
-----
    python run_experiment.py --axis sentiment --model gpt2 --layer 8
    python run_experiment.py --axis toxicity  --model gpt2 --layer 8 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cgsae_core import (  # noqa: E402
    RunConfig, build_sae, train_sae, evaluate_sae, select_latents,
    latent_class_auc, latent_selectivity, encode_to_cpu, set_seed, auc, CGSAE,
)
from cgsae_model import (  # noqa: E402
    HookedLM, Judge, collect_token_activations, standardise,
    make_sae_edit, make_direction_edit, diff_in_means_direction, probe_direction,
)
from cgsae_data import load_axis, load_ppl_corpus, refusal_rate  # noqa: E402


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--axis", default="sentiment", choices=["sentiment", "toxicity", "refusal"])
    p.add_argument("--model", default="gpt2")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--expansion", type=int, default=8)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--n-docs", type=int, default=4000)
    p.add_argument("--max-tokens", type=int, default=400_000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--n-select", type=int, default=64)
    p.add_argument("--n-selects", type=int, nargs="+", default=None,
                   help="sweep the number of latents intervened on. Needed for a fair "
                        "block comparison: zeroing CG-SAE's 614-wide B-block against a "
                        "64-latent post-hoc selection would reward intervention size, "
                        "not method.")
    p.add_argument("--save-generations", action="store_true",
                   help="dump example completions per arm for qualitative inspection")
    p.add_argument("--n-prompts", type=int, default=120)
    p.add_argument("--gen-tokens", type=int, default=32)
    p.add_argument("--ppl-docs", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--tag-fracs", type=float, nargs="+",
                   default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--dir-alphas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--lambda-cg", type=float, default=3.0)
    p.add_argument("--pen-mode", default="contrast",
                   choices=["contrast","ratio","mean","sq_mean"],
                   help="cross-group penalty form; sq_mean is v1 (inert under sparsity)")
    p.add_argument("--pen-modes", nargs="+", default=None,
                   help="calibration: sweep penalty forms")
    p.add_argument("--lambda-cgs", type=float, nargs="+", default=None,
                   help="calibration: sweep the penalty weight instead of the tag fraction")
    p.add_argument("--skip-steering", action="store_true",
                   help="train and score SAEs only; no generation (fast calibration)")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"],
                   help="fp16 for models above ~1B so the activation table and the "
                        "model both fit on a 16GB P100")
    p.add_argument("--probe-layers", type=int, nargs="+", default=None,
                   help="report the linear ceiling at each layer, then exit (layer choice)")
    p.add_argument("--layer-scan", action="store_true",
                   help="scan EVERY layer for both linear decodability and actual steering "
                        "efficacy, then exit. Probe AUC alone is a poor layer criterion: it "
                        "tends to rise in late layers because the residual is already "
                        "committing to the next token, which is a readout property rather "
                        "than a steerable representation - and an edit at the last layer "
                        "has no downstream computation left to act on.")
    p.add_argument("--scan-alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0],
                   help="steering strengths used by --layer-scan")
    p.add_argument("--out", default="results")
    p.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    return p.parse_args()


def apply_smoke(a: argparse.Namespace) -> None:
    a.n_docs, a.max_tokens, a.epochs = 200, 20_000, 2
    a.n_prompts, a.ppl_docs, a.gen_tokens = 16, 20, 12
    a.seeds, a.tag_fracs = [0], [0.10, 0.50]
    a.alphas, a.dir_alphas = [0.0, 1.0], [0.0, 2.0]
    a.expansion = 4


# ---------------------------------------------------------------------------
def steering_sweep(lm, axis, judge, arm_name, seed, make_edit, alphas, prompts,
                   control_prompts, ppl_texts, layer, gen_tokens, base,
                   gen_sink=None):
    """Sweep intervention strength, recording effect and collateral at each point.

    `gen_sink`, when given, collects a few completions per operating point. A judge score
    cannot distinguish "steered to positive sentiment" from "degenerated into repeated
    positive words", and the direction baselines reach +226% perplexity at full strength -
    so the qualitative check is not optional.
    """
    rows = []
    for alpha in alphas:
        edit = make_edit(alpha)
        gen = lm.generate(prompts, layer, edit, max_new_tokens=gen_tokens)
        ctrl_gen = lm.generate(control_prompts, layer, edit, max_new_tokens=gen_tokens)
        ppl = lm.perplexity(ppl_texts, layer, edit)
        if gen_sink is not None:
            for p, g in list(zip(prompts, gen))[:4]:
                gen_sink.append({"arm": arm_name, "seed": seed, "alpha": alpha,
                                 "prompt": p, "completion": g})

        if axis == "refusal":
            eff = refusal_rate(gen)
            ctrl_eff = refusal_rate(ctrl_gen)
        else:
            eff = float(np.mean(judge.score(gen)))
            ctrl_eff = float(np.mean(judge.score(ctrl_gen)))

        rows.append({
            "arm": arm_name, "seed": seed, "alpha": alpha,
            "unwanted_score": eff,
            "unwanted_score_base": base["eff"],
            "effect": base["eff"] - eff,                     # reduction in unwanted behaviour
            "control_score": ctrl_eff,
            "control_score_base": base["ctrl"],
            "collateral": abs(ctrl_eff - base["ctrl"]),      # damage on the wanted class
            "ppl": ppl,
            "ppl_base": base["ppl"],
            "ppl_pct": 100.0 * (ppl / base["ppl"] - 1.0),
        })
        print(f"    {arm_name:22s} a={alpha:<5.2f} effect={rows[-1]['effect']:+.3f} "
              f"collat={rows[-1]['collateral']:.3f} ppl={ppl:.2f} "
              f"({rows[-1]['ppl_pct']:+.1f}%)", flush=True)
    return rows


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke(args)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    cfg = RunConfig(
        model_name=args.model, layer=args.layer, axis=args.axis,
        n_docs=args.n_docs, max_train_tokens=args.max_tokens,
        expansion=args.expansion, topk=args.topk, sae_epochs=args.epochs,
        lambda_cg=args.lambda_cg, pen_mode=args.pen_mode, n_select=args.n_select,
        tag_fracs=tuple(args.tag_fracs), alphas=tuple(args.alphas),
        seeds=tuple(args.seeds), out_dir=args.out, device=device,
    )
    print(cfg.to_json(), flush=True)

    # -- data ---------------------------------------------------------------
    print("\n[1/5] loading data", flush=True)
    data = load_axis(args.axis, n_docs=args.n_docs, n_prompts=args.n_prompts)
    ppl_texts = load_ppl_corpus(n_docs=args.ppl_docs)
    print(f"  docs={len(data['texts'])} prompts={len(data['prompts'])} "
          f"ctrl={len(data['control_prompts'])} ppl_docs={len(ppl_texts)}", flush=True)

    # -- model + activations -------------------------------------------------
    print("\n[2/5] loading model and collecting per-token activations", flush=True)
    lm = HookedLM(args.model, device=device,
                  dtype=torch.float16 if args.dtype == "fp16" else torch.float32)
    print(f"  {args.model}: {lm.n_layers} layers, d_model={lm.d_model}, "
          f"layer={args.layer}, dtype={args.dtype}")

    # -- full layer scan ------------------------------------------------------
    # Every layer, two criteria: how linearly decodable the concept is, and how well an
    # intervention there actually works. Steering uses difference-in-means, which needs no
    # SAE training, so the whole scan costs about a minute per layer.
    if args.layer_scan:
        from sklearn.linear_model import LogisticRegression as _LRS
        judge_s = None if args.axis == "refusal" else Judge(args.axis, device=device)
        base_gen = lm.generate(data["prompts"], 0, None, max_new_tokens=args.gen_tokens)
        base_eff = (refusal_rate(base_gen) if args.axis == "refusal"
                    else float(np.mean(judge_s.score(base_gen))))
        base_ppl = lm.perplexity(ppl_texts, 0, None)
        print(f"  unpatched: unwanted={base_eff:.3f} ppl={base_ppl:.2f}", flush=True)

        rows = []
        for L in range(lm.n_layers):
            a_l, y_l, _ = collect_token_activations(
                lm, data["texts"], data["labels"], L, max_seq_tokens=128,
                max_tokens=min(args.max_tokens, 120_000))
            a_s, mu_l, sg_l = standardise(a_l)
            n_l = int(0.8 * len(a_s))
            y_bin = (y_l.numpy() == 0).astype(int)
            pr = _LRS(max_iter=1000).fit(a_s[:n_l].numpy(), y_bin[:n_l])
            p_auc = auc(pr.decision_function(a_s[n_l:].numpy()), y_bin[n_l:])

            vec = (diff_in_means_direction(a_s, y_l).to(device) * sg_l.to(device))
            scale = float(a_s.norm(dim=-1).mean()) * float(sg_l) / 10.0
            best = {"effect": 0.0, "ppl_pct": 0.0, "alpha": 0.0}
            for al in args.scan_alphas:
                ed = make_direction_edit(vec, al, scale)
                g = lm.generate(data["prompts"], L, ed, max_new_tokens=args.gen_tokens)
                eff = (refusal_rate(g) if args.axis == "refusal"
                       else float(np.mean(judge_s.score(g))))
                pp = lm.perplexity(ppl_texts, L, ed)
                pct = 100.0 * (pp / base_ppl - 1.0)
                # Efficiency, not raw effect: reduction per unit of fluency spent. A layer
                # that only "works" by wrecking the model is not a good place to intervene.
                effc = (base_eff - eff) / max(pct, 1.0)
                if effc > best.get("efficiency", -9e9):
                    best = {"effect": base_eff - eff, "ppl_pct": pct,
                            "alpha": al, "efficiency": effc}
            rows.append({"layer": L, "probe_auc": p_auc, **best})
            print(f"  L{L:02d}  probe_auc={p_auc:.4f}  best_effect={best['effect']:+.3f} "
                  f"@a={best['alpha']:g} ppl={best['ppl_pct']:+.1f}% "
                  f"eff/ppl={best['efficiency']:+.4f}", flush=True)
            del a_l, a_s
            torch.cuda.empty_cache()

        df = pd.DataFrame(rows)
        tag = f"{args.axis}_{args.model.split('/')[-1]}"
        df.to_csv(f"{args.out}/layer_scan_{tag}.csv", index=False)
        b_probe = df.loc[df.probe_auc.idxmax()]
        b_steer = df.loc[df.efficiency.idxmax()]
        print(f"\n  best probe AUC     : layer {int(b_probe.layer)} ({b_probe.probe_auc:.4f})")
        print(f"  best steer efficiency: layer {int(b_steer.layer)} "
              f"(effect {b_steer.effect:+.3f} at {b_steer.ppl_pct:+.1f}% ppl)")
        if int(b_probe.layer) != int(b_steer.layer):
            print("  -> decodability and interventional efficacy peak at DIFFERENT layers; "
                  "pick the layer by steering, not by probe AUC.")
        return

    # -- layer selection ------------------------------------------------------
    # Which layer carries the concept most linearly? v1 used layer 8 without justifying it.
    # This is the cheap, principled answer, and it is the number to cite when asked.
    if args.probe_layers:
        from sklearn.linear_model import LogisticRegression as _LRL
        rows = []
        for L in args.probe_layers:
            a_l, y_l, _ = collect_token_activations(
                lm, data["texts"], data["labels"], L,
                max_seq_tokens=128, max_tokens=min(args.max_tokens, 200_000))
            a_s, _, _ = standardise(a_l)
            n_l = int(0.8 * len(a_s))
            y_bin = (y_l.numpy() == 0).astype(int)
            pr = _LRL(max_iter=2000).fit(a_s[:n_l].numpy(), y_bin[:n_l])
            a_auc = auc(pr.decision_function(a_s[n_l:].numpy()), y_bin[n_l:])
            rows.append({"layer": L, "probe_auc": a_auc, "n_tokens": len(a_s)})
            print(f"  layer {L:3d}: per-token linear probe AUC = {a_auc:.4f}", flush=True)
            del a_l, a_s
            torch.cuda.empty_cache()
        df = pd.DataFrame(rows)
        best = df.loc[df.probe_auc.idxmax()]
        print(f"\n  best layer = {int(best.layer)} (AUC {best.probe_auc:.4f})")
        df.to_csv(f"{args.out}/layer_probe_{args.axis}_"
                  f"{args.model.split('/')[-1]}.csv", index=False)
        return

    acts, labels, docs = collect_token_activations(
        lm, data["texts"], data["labels"], args.layer,
        max_seq_tokens=128, max_tokens=args.max_tokens)
    print(f"  activation table: {tuple(acts.shape)}  "
          f"({int((labels == 1).sum())} wanted / {int((labels == 0).sum())} unwanted tokens)")

    acts_std, mu, sigma = standardise(acts)
    acts_std = acts_std.to(device)
    labels_d = labels.to(device)
    mu, sigma = mu.to(device), sigma.to(device)

    n = acts_std.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    A_tr, L_tr = acts_std[tr.to(device)], labels_d[tr.to(device)]
    A_va, L_va = acts_std[va.to(device)], labels_d[va.to(device)]
    A_te, L_te = acts_std[te.to(device)], labels_d[te.to(device)]
    print(f"  split: train={len(tr)} val={len(va)} test={len(te)}")

    # -- ceiling diagnostic ---------------------------------------------------
    # How much of the concept is linearly present in the raw per-token residual at all?
    # No SAE latent can carry more class information than this. If it sits near chance the
    # per-token labelling is the problem, not the autoencoder - and every downstream number
    # would be uninterpretable. v1 never checked this.
    print("\n[2b] linear ceiling on the raw activations", flush=True)
    from sklearn.linear_model import LogisticRegression as _LR
    _n = min(80_000, A_tr.shape[0])
    # Fit with the *unwanted* class as the positive label so decision_function is already
    # oriented the same way as the AUC target; the first calibration run had these opposed
    # and reported 0.225 where the true ceiling was 1 - 0.225 = 0.775.
    _y_tr = (L_tr[:_n].cpu().numpy() == 0).astype(int)
    _probe = _LR(max_iter=2000).fit(A_tr[:_n].cpu().numpy(), _y_tr)
    _score = _probe.decision_function(A_te.cpu().numpy())
    raw_probe_auc = auc(_score, (L_te.cpu().numpy() == 0).astype(int))
    print(f"  per-token linear probe AUC (unwanted vs wanted) = {raw_probe_auc:.4f}")
    if raw_probe_auc < 0.60:
        print("  !! WARNING: the concept is barely linearly decodable per token. "
              "SAE latent AUCs are capped near this; interpret arms with care.", flush=True)

    judge = None if args.axis == "refusal" else Judge(args.axis, device=device)

    # -- unpatched baseline --------------------------------------------------
    base = {"eff": float("nan"), "ctrl": float("nan"), "ppl": float("nan")}
    if not args.skip_steering:
        print("\n[3/5] unpatched baseline", flush=True)
        base_gen = lm.generate(data["prompts"], args.layer, None,
                               max_new_tokens=args.gen_tokens)
        base_ctrl = lm.generate(data["control_prompts"], args.layer, None,
                                max_new_tokens=args.gen_tokens)
        base_ppl = lm.perplexity(ppl_texts, args.layer, None)
        if args.axis == "refusal":
            base = {"eff": refusal_rate(base_gen),
                    "ctrl": refusal_rate(base_ctrl), "ppl": base_ppl}
        else:
            base = {"eff": float(np.mean(judge.score(base_gen))),
                    "ctrl": float(np.mean(judge.score(base_ctrl))), "ppl": base_ppl}
        print(f"  unwanted={base['eff']:.3f}  control={base['ctrl']:.3f}  ppl={base['ppl']:.2f}")

    d_latent = args.expansion * lm.d_model
    metric_rows, steer_rows, sel_rows = [], [], []
    gen_rows = [] if args.save_generations else None

    # -- arms ----------------------------------------------------------------
    for seed in args.seeds:
        print(f"\n[4/5] seed {seed}", flush=True)
        set_seed(seed)

        if args.lambda_cgs:
            # Calibration mode: hold the tag fraction fixed and sweep the penalty weight,
            # to find the range where the penalty bites without destroying reconstruction.
            arms = [("vanilla", None)] + [("cgsae", tf) for tf in args.tag_fracs]
        else:
            arms = [("vanilla", None), ("scar", None)] + \
                   [("cgsae", tf) for tf in args.tag_fracs]

        lambdas = args.lambda_cgs if args.lambda_cgs else [args.lambda_cg]
        pen_modes = args.pen_modes if args.pen_modes else [args.pen_mode]
        combos = [(lam, pm) for pm in pen_modes for lam in lambdas]

        for kind, tf in arms:
            for lam, pm in (combos if kind == "cgsae" else [(args.lambda_cg, args.pen_mode)]):
                cfg.lambda_cg, cfg.pen_mode = lam, pm
                tag = f"{kind}" + (f"@{tf:.2f}" if tf is not None else "")
                if kind == "cgsae":
                    if args.pen_modes:
                        tag += f"_{pm}"
                    if args.lambda_cgs:
                        tag += f"_lam{lam:g}"
                print(f"  training {tag} (d_latent={d_latent}, k={args.topk}, "
                      f"lambda_cg={lam})", flush=True)
                set_seed(seed)
                sae = build_sae(kind, lm.d_model, d_latent, args.topk, cfg,
                                tag_frac=tf).to(device)
                train_sae(sae, A_tr, L_tr, cfg, val_acts=A_va, val_labels=L_va, verbose=True)

                m = evaluate_sae(sae, A_te, L_te)
                # Encode once to CPU (subsampled) and reuse: at d_latent=12288 a full
                # on-GPU latent matrix plus any elementwise copy exceeds the P100.
                z_va, z_rows = encode_to_cpu(sae, A_va, max_rows=40_000)
                L_va_s = L_va[z_rows.to(L_va.device)].cpu()
                lat_auc = latent_class_auc(z_va, L_va_s)
                lat_sel, lat_freq, eligible = latent_selectivity(z_va, L_va_s)
                # Sparsity-robust view: among latents that actually fire, how selective is
                # the most selective one, and how many are strongly class-biased?
                elig_sel = lat_sel[eligible] if eligible.any() else np.array([0.5])
                m.update({"arm": tag, "kind": kind, "seed": seed, "tag_frac": tf,
                          "lambda_cg": lam, "pen_mode": cfg.pen_mode,
                          "d_latent": d_latent, "k": args.topk,
                          "raw_probe_auc": raw_probe_auc,
                          "best_latent_auc": float(np.nanmax(np.abs(lat_auc - 0.5)) + 0.5),
                          "n_latents_auc_gt_60": int((lat_auc > 0.60).sum()),
                          "best_selectivity": float(np.nanmax(elig_sel)),
                          "n_sel_gt_70": int((elig_sel > 0.70).sum()),
                          "n_sel_gt_90": int((elig_sel > 0.90).sum()),
                          "n_eligible": int(eligible.sum()),
                          "n_dead_latents": int((lat_freq == 0).sum())})
                metric_rows.append(m)
                print(f"    -> fvu={m['fvu']:.4f} l0={m['l0']:.1f} "
                      f"leak_norm={m.get('norm_leakage', float('nan')):.4f} "
                      f"best_sel={m['best_selectivity']:.3f} "
                      f"n_sel>.7={m['n_sel_gt_70']} n_sel>.9={m['n_sel_gt_90']} "
                      f"live={m['n_eligible']} dead={m['n_dead_latents']}")

                # Selection variants. `free` = post-hoc over all latents (what every arm
                # gets, and what v1 actually did). `group` = restricted to the B-block, the
                # only variant that uses the grouping. Comparing the two isolates whether
                # the training-time partition adds anything beyond post-hoc ranking.
                n_sel_list = args.n_selects or [args.n_select]
                variants = [(f"free@{n}", select_latents(z_va, L_va_s, n, None))
                            for n in n_sel_list if n <= d_latent]
                if isinstance(sae, CGSAE) and len(sae.b_idx) >= args.n_select:
                    variants.append((f"group@{args.n_select}",
                                     select_latents(z_va, L_va_s, args.n_select, sae.b_idx)))
                if isinstance(sae, CGSAE):
                    # The method as actually proposed: zero the entire B-block **by
                    # address**. No ranking, no labelled probe set at intervention time -
                    # the unwanted subspace is a fixed index range fixed before training.
                    # This is the only arm with the operational property that distinguishes
                    # CG-SAE from post-hoc selection (SAE-SSV) and it makes tau a dose:
                    # how much of the dictionary must be surrendered to remove the concept.
                    b_block = sae.b_idx.cpu().numpy()
                    variants.append(("block", b_block))
                    # Size-matched null: a random block of the same width, to separate
                    # "this region carries the concept" from "we zeroed a lot of latents".
                    variants.append(("blockrand", np.random.RandomState(seed + 977).choice(
                        d_latent, len(b_block), replace=False)))

                for vname, sel in variants:
                    sel_rows.append({
                        "arm": tag, "seed": seed, "selection": vname,
                        "n_selected": int(len(sel)),
                        "mean_auc": float(np.nanmean(lat_auc[sel])),
                        "max_auc": float(np.nanmax(lat_auc[sel])),
                        "mean_selectivity": float(np.nanmean(lat_sel[sel])),
                        "frac_alive": float(np.mean(lat_freq[sel] > 0)),
                        "n_in_bgroup": int(np.isin(sel, sae.b_idx.cpu().numpy()).sum())
                        if isinstance(sae, CGSAE) else 0,
                    })
                    if not args.skip_steering:
                        steer_rows += steering_sweep(
                            lm, args.axis, judge, f"{tag}/{vname}", seed,
                            lambda a, s=sae, idx=sel: make_sae_edit(s, idx, a, mu, sigma),
                            args.alphas, data["prompts"], data["control_prompts"],
                            ppl_texts, args.layer, args.gen_tokens, base, gen_rows)

                if not args.skip_steering:
                    # null control: random latents of the same count, same SAE
                    rng = np.random.RandomState(seed)
                    rand_sel = rng.choice(d_latent, args.n_select, replace=False)
                    steer_rows += steering_sweep(
                        lm, args.axis, judge, f"{tag}/random", seed,
                        lambda a, s=sae, idx=rand_sel: make_sae_edit(s, idx, a, mu, sigma),
                        args.alphas, data["prompts"], data["control_prompts"], ppl_texts,
                        args.layer, args.gen_tokens, base, gen_rows)

                del sae
                torch.cuda.empty_cache()

        if args.skip_steering:
            continue

        # -- non-SAE direction baselines --------------------------------------
        print("  direction baselines", flush=True)
        scale = float(A_tr.norm(dim=-1).mean().item()) * float(sigma.item()) / 10.0
        dirs = {
            "diff_in_means": diff_in_means_direction(A_tr, L_tr).to(device) * sigma,
            "probe": probe_direction(A_tr, L_tr).to(device) * sigma,
        }
        g = torch.Generator(device="cpu").manual_seed(seed)
        dirs["random_dir"] = (torch.randn(lm.d_model, generator=g).to(device) * sigma)

        for dname, vec in dirs.items():
            steer_rows += steering_sweep(
                lm, args.axis, judge, dname, seed,
                lambda a, v=vec: make_direction_edit(v, a, scale),
                args.dir_alphas, data["prompts"], data["control_prompts"], ppl_texts,
                args.layer, args.gen_tokens, base, gen_rows)

    # -- save ----------------------------------------------------------------
    print("\n[5/5] writing results", flush=True)
    tagname = f"{args.axis}_{args.model.split('/')[-1]}_L{args.layer}"
    pd.DataFrame(metric_rows).to_csv(f"{args.out}/sae_metrics_{tagname}.csv", index=False)
    pd.DataFrame(steer_rows).to_csv(f"{args.out}/steering_{tagname}.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(f"{args.out}/selection_{tagname}.csv", index=False)
    if gen_rows:
        pd.DataFrame(gen_rows).to_csv(f"{args.out}/generations_{tagname}.csv", index=False)
    with open(f"{args.out}/config_{tagname}.json", "w") as f:
        json.dump({"args": vars(args), "baseline": base,
                   "d_latent": d_latent, "n_tokens": int(acts.shape[0])}, f, indent=2)
    print(f"  wrote {args.out}/*_{tagname}.csv")
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
