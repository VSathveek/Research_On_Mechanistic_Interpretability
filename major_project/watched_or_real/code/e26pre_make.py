# Builds e26pre.py from e21.py: inference only, natural model, no generation, no test-set behaviour.
# Freezes, before any E26 training: the natural refusal direction per layer, the attack-shift PCs fit on the training wrapper
# families (role-play templates excluded), and the redundancy score k* (how many attack PCs must be patched to the plain-request
# value at the window layers before the late-layer refusal push falls by >= 80%), with rank-matched random and 1-d controls.
import sys
SRC, OUTF = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
head = s[s.index("import os;"):s.index("# ---------------- training (BSCT / ACT) ----------------")]
assert head.count('METHOD = "bsct"') == 1
head = head.replace('METHOD = "bsct"', 'METHOD = "pre"').replace("e21_crossed_", "e26pre_crossed_")
tail = r'''
# ======================= E26-pre =======================
ALLL = list(range(1, NL + 1)); late = [l for l in ALLL if l >= SEL + 3]
FIC_TR = [TRAIN_FIC[i] for i in (0, 2, 4, 5)]                     # role-play templates (1: screenplay villain, 3: tabletop NPC) held out
NG = len(train_goals); FIT = list(range(NG // 2)); EVL = list(range(NG // 2, NG))   # fit PCs on the first half of the training goals
Hh = last_states([chat(g) for g in train_goals], ALLL); Hb = last_states([chat(b) for b in ben_tasks[:NG]], ALLL)
RD = {l: torch.tensor(unit((Hh[l].mean(0) - Hb[l].mean(0)).numpy())) for l in ALLL}
RG = {l: float((Hh[l] @ RD[l]).mean() - (Hb[l] @ RD[l]).mean()) for l in ALLL}
PAIRS = [(gi, w) for gi in range(NG) for w in FIC_TR]
Hf = last_states([chat(w.format(g=low(train_goals[gi]))) for gi, w in PAIRS], ALLL)
GI = np.array([gi for gi, _ in PAIRS]); mf = np.isin(GI, FIT); me = np.isin(GI, EVL)
DIFF = {l: Hf[l] - Hh[l][GI] for l in ALLL}
KMAX = 128 if not SMOKE else 16
V = {l: torch.linalg.svd(DIFF[l][mf], full_matrices=False)[2][:KMAX] for l in ALLL}                     # [k, D], uncentred
def push(H):                                                       # late-layer refusal push of fiction prompts, units of the refusal gap
    return float(np.mean([float(((H[l] - Hh[l][GI[me]]) @ RD[l]).mean() / RG[l]) for l in late]))
base = push({l: Hf[l][me] for l in late}); print(f"[pre] natural late refusal push on held-out training goals: {base:.3f}")
HOOK = []
def patch(layers, basis_of):                                       # set the decision-token projection on the basis to the plain-request value
    for l in layers:
        Bm = basis_of(l)
        if Bm is None: continue
        Bm = Bm.float().cuda(0) if torch.cuda.is_available() else Bm.float()
        tgt = Hh[l][GI[me]]
        def hook(m, i, o, l=l, Bm=Bm, tgt=tgt):
            h = o[0] if isinstance(o, tuple) else o
            h = h.clone(); x = h[:, -1].float(); B_ = Bm.to(x.device); t_ = CURT[l].to(x.device)
            h[:, -1] = (x - ((x - t_) @ B_.T) @ B_).to(h.dtype)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        HOOK.append(LAYMOD[l - 1].register_forward_hook(hook))
LAYMOD = next(m for n, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and n.endswith("layers") and len(m) == NL and "vision" not in n)
TX = [chat(w.format(g=low(train_goals[gi]))) for gi, w in PAIRS]; TX = [t for t, e in zip(TX, me) if e]; TGI = GI[me]
CURT = {}
@torch.no_grad()
def run_push(layers, basis_of):
    patch(layers, basis_of); out = {l: [] for l in late}
    for i in range(0, len(TX), 8):
        for l in layers: CURT[l] = Hh[l][TGI[i:i + 8]]
        enc = tok(TX[i:i + 8], return_tensors="pt", padding=True, add_special_tokens=False).to(EMB)
        hs = model(**enc, output_hidden_states=True).hidden_states
        for l in late: out[l].append(hs[l][:, -1].float().cpu())
    for h in HOOK: h.remove()
    HOOK.clear(); return push({l: torch.cat(out[l]) for l in late})
def rand_basis(k, seed):
    return torch.linalg.qr(torch.tensor(np.random.default_rng(seed).standard_normal((Hh[SEL].shape[1], k)).astype(np.float32)))[0].T
ROWS = []
KS_ = [k for k in [1, 2, 4, 8, 16, 32, 64, 128] if k <= KMAX]
for lay_name, layers in [("win", LW), ("sel", [SEL])]:
    for k in KS_:
        ROWS.append(dict(layers=lay_name, arm="pcs", k=k, push=run_push(layers, lambda l, k=k: V[l][:k])))
        ROWS.append(dict(layers=lay_name, arm="rand", k=k, push=np.mean([run_push(layers, lambda l, k=k, j=j: rand_basis(k, 100 * k + j + l)) for j in range(3)])))
    ROWS.append(dict(layers=lay_name, arm="refusal", k=1, push=run_push(layers, lambda l: RD[l][None])))
    ROWS.append(dict(layers=lay_name, arm="real", k=1, push=run_push(layers, lambda l: torch.tensor(unit(DR[l]))[None] if l in DR else None)))
    ROWS.append(dict(layers=lay_name, arm="full", k=-1, push=run_push(layers, lambda l: torch.eye(Hh[SEL].shape[1]))))
    print(f"[pre] {lay_name} done ({time.time()-t0:.0f}s)")
T = pd.DataFrame(ROWS); T["frac_left"] = T.push / base; T.to_csv(f"{OUT}/e26pre_patch_curve.csv", index=False); print(T.round(3).to_string(index=False))
VERS = {}
for m_ in ["torch", "transformers", "peft", "bitsandbytes"]:
    try: VERS[m_] = __import__(m_).__version__
    except Exception: pass
def kstar(lay):
    q = T[(T.layers == lay) & (T.arm == "pcs")].sort_values("k"); ok = q[q.frac_left <= 0.2]
    return int(ok.k.iloc[0]) if len(ok) else 10 ** 6                # 10^6 = not reached within KMAX
# refusal-weighted participation ratio at SEL (fit half): share of the push carried by each PC
D_ = DIFF[SEL][mf]; s_ = np.array([float((D_ @ V[SEL][i]).mean() * (V[SEL][i] @ RD[SEL])) for i in range(KMAX)])
PR = float(np.abs(s_).sum() ** 2 / (s_ ** 2).sum())
EN = {l: (np.cumsum((DIFF[l][mf] @ V[l].T).pow(2).mean(0).numpy()) / float(DIFF[l][mf].pow(2).sum(-1).mean())) for l in [SEL]}
SC = dict(model=MODEL, sel=SEL, base_push=base, kstar_win=kstar("win"), kstar_sel=kstar("sel"), refusal_pr=PR,
          refusal_1d_win_frac_left=float(T[(T.layers == "win") & (T.arm == "refusal")].frac_left.iloc[0]),
          real_1d_win_frac_left=float(T[(T.layers == "win") & (T.arm == "real")].frac_left.iloc[0]),
          full_win_frac_left=float(T[(T.layers == "win") & (T.arm == "full")].frac_left.iloc[0]),   # positional redundancy: push left after patching the whole decision-token state
          energy_sel_k={k: float(EN[SEL][k - 1]) for k in KS_}, n_fit=int(mf.sum()), n_eval=int(me.sum()),
          versions=VERS)
print("[pre] SCORES", json.dumps(SC))
json.dump(SC, open(f"{OUT}/e26pre_scores.json", "w"), indent=1)
np.savez_compressed(f"{OUT}/e26pre_bases.npz", RD=np.stack([RD[l].numpy() for l in ALLL]).astype(np.float16),
                    RG=np.array([RG[l] for l in ALLL]), V32=np.stack([V[l][:32].numpy() for l in ALLL]).astype(np.float16),
                    fic_tr=np.array(json.dumps(FIC_TR)))
print("DONE", f"{time.time()-t0:.0f}s")
'''
s2 = "# E26-pre -- frozen inference-only bases and redundancy score before E26 training.\n" + head + tail
open(OUTF, "w", encoding="utf-8").write(s2); compile(s2, "e26pre", "exec"); print("ok", len(s2))
