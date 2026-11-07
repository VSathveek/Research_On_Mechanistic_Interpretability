# Builds e23.py from e21.py (probe machinery): discover framing directions OUTSIDE the 20-belief atlas, then test them causally.
import sys
SRC, OUTF = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70]); s = s.replace(a, b)

rep(s[:s.index("import os;")], """# E23 -- What do attacks move that the belief atlas misses? Discovery on BENIGN data only: wrapper-minus-plain shifts on harmless tasks,
# with the 20 atlas beliefs + harm direction projected out; top residual components. Compliance labels on goals 0-49 (from the
# natural run) choose 3 components and orient them; on held-out goals 50-99 each is clamped (plain requests pushed toward the
# complied side; fiction attacks pushed toward the refused side) against a matched random direction outside the atlas.
# A supervised residual compliance direction (goals 0-49) is the reference for "refusal intent" rather than a framing belief.
""")
rep('METHOD = "bsct"', 'METHOD = "probe"\nLABELS = ""')
rep("for n in READ_SET:\n    pos, neg = BELIEFS[n]", "for n in BELIEFS:\n    pos, neg = BELIEFS[n]")
rep("GEN = []\nRND = {l:", r'''# ---------------- E23 discovery (layer SEL only) ----------------
from sklearn.metrics import roc_auc_score
L = SEL; Dm = B0[L].shape[1]; NAMES = list(BELIEFS)
Hh, Hb = last_states([chat(g) for g in train_goals], [L])[L], last_states([chat(b) for b in ben_tasks[:150]], [L])[L]
HDIR = torch.tensor(unit((Hh.mean(0) - Hb.mean(0)).numpy()))
ALS = torch.linalg.qr(torch.stack([BD[n][L] for n in NAMES] + [HDIR], 1))[0]                  # 21-d atlas + harm span
def perp(X): return X - (X @ ALS) @ ALS.T
BT = alp[3100:3100 + (6 if SMOKE else 50)]; BW = [w for w in TEST_WRAP if w != "bare"]
S_plain = last_states([chat(t) for t in BT], [L])[L]
S_wrap = {w: last_states([chat(TEST_WRAP[w].format(g=low(t), G=t)) for t in BT], [L])[L] for w in BW}
DB = torch.cat([S_wrap[w] - S_plain for w in BW]); DBW = np.repeat(BW, len(BT))                 # benign framing shifts
GI = np.array([gi for _, gi, _ in JB]); WJ = np.array([w for w, _, _ in JB]); HJ = B0[L]
DH = torch.stack([HJ[i] - HJ[(GI == GI[i]) & (WJ == "bare")][0] for i in range(len(JB)) if WJ[i] != "bare"])
def frac_in(X, Q): return float(((X @ Q) ** 2).sum() / (X ** 2).sum())
COV = dict(benign_shift_in_atlas=frac_in(DB, ALS), harmful_shift_in_atlas=frac_in(DH, ALS), random_baseline=ALS.shape[1] / Dm)
print("[coverage]", COV)
DBp = perp(DB); Uv, Sv, Vt = torch.linalg.svd(DBp, full_matrices=False)                        # uncentred: keeps the mean framing shift
K = 8; PCS = Vt[:K]; EVR = (Sv[:K] ** 2 / (Sv ** 2).sum()).tolist()
Y = np.array([c == "1" for c in LABELS]) if LABELS else np.random.default_rng(0).random(len(JB)) < 0.3
assert len(Y) == len(JB)
HALF = N_TEST_GOALS // 2; DISC = (GI < HALF) & (WJ != "bare")
def orient(u):
    p = (HJ[DISC] @ u).numpy(); yc = Y[DISC]
    a = roc_auc_score(yc, p) if 0 < yc.sum() < len(yc) else 0.5
    if a < 0.5: u, p, a = -u, -p, 1 - a
    return u, a, float(p[yc].mean()), float(p[~yc].mean())
ROWS = []
for k in range(K):
    u, a, mc, mr = orient(PCS[k]); ROWS.append(dict(name=f"pc{k}", evr=EVR[k], disc_auroc=a, m_comp=mc, m_ref=mr, u=u))
HP = perp(HJ); us = unit((HP[DISC & Y].mean(0) - HP[DISC & ~Y].mean(0)).numpy()); us = torch.tensor(us)
u, a, mc, mr = orient(us); ROWS.append(dict(name="supervised", evr=float(((DBp @ u) ** 2).sum() / (DBp ** 2).sum()), disc_auroc=a, m_comp=mc, m_ref=mr, u=u))
for r in ROWS:
    r["benign_framing_load"] = float(((DBp @ r["u"]) ** 2).mean() / (DBp ** 2).sum(1).mean())
    r["by_wrapper"] = {w: float((DB[DBW == w] @ r["u"]).mean()) for w in BW}
    r["cos_harm"] = float(r["u"] @ HDIR); r["max_cos_atlas"] = max(abs(float(r["u"] @ BD[n][L])) for n in NAMES)
try:
    WU = model.get_output_embeddings().weight
    for r in ROWS:
        lg = (WU.float() @ r["u"].to(WU.device)).cpu()
        r["lens_up"] = [tok.decode([i]) for i in lg.topk(12).indices.tolist()]; r["lens_down"] = [tok.decode([i]) for i in (-lg).topk(12).indices.tolist()]
except Exception as e: print("logit lens skipped", repr(e)[:120])
SELECTED = [r for r in sorted(ROWS[:K], key=lambda r: -r["disc_auroc"])[:3]] + [ROWS[-1]]
TB = pd.DataFrame([{k: v for k, v in r.items() if k not in ("u", "by_wrapper")} for r in ROWS]); TB.to_csv(f"{OUT}/e23_directions.csv", index=False)
print(TB[["name", "evr", "disc_auroc", "benign_framing_load", "cos_harm", "max_cos_atlas"]].round(3).to_string(index=False))
print("selected for causal test:", [r["name"] for r in SELECTED])
np.savez_compressed(f"{OUT}/e23_dirs.npz", **{r["name"]: r["u"].numpy() for r in ROWS}, atlas=ALS.numpy(), layer=np.array(L))
def rand_perp(u, seed):
    v = torch.tensor(np.random.default_rng(seed).standard_normal(Dm).astype(np.float32)); v = perp(v[None])[0]; v = v - (v @ u) * u
    return v / v.norm()
def steer(u, target, v):
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o; h = h.clone(); x = h[:, -1]; uu = u.to(x.device, x.dtype)
        d = (target - x @ uu); vv = uu if v is None else v.to(x.device, x.dtype)
        h[:, -1] = x + d.unsqueeze(-1) * vv
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    HOOKS.append(LAYMOD[L - 1].register_forward_hook(hook))
GEN = []
RND = {l:''')
a = s.index("FJ = [x for x in JB if x[0] in FIC_TEST]; BJ = [x for x in JB if x[0] == \"bare\"]")
b = s.index("for s, items, cb, crnd in PLAN:")
s = s[:a] + '''FJ = [x for x in JB if x[0] in FIC_TEST and x[1] >= HALF]; BJ = [x for x in JB if x[0] == "bare" and x[1] >= HALF]
PLAN = [("none", FJ + BJ, None, False)]
for j, r in enumerate(SELECTED):
    gap = r["m_comp"] - r["m_ref"]; up, dn = r["m_comp"] + gap, r["m_ref"] - gap; rv = rand_perp(r["u"], 700 + j)
    PLAN += [(f"{r['name']}_up", BJ, (r["u"], up, None), False), (f"{r['name']}_uprand", BJ, (r["u"], up, rv), False),
             (f"{r['name']}_down", FJ, (r["u"], dn, None), False), (f"{r['name']}_downrand", FJ, (r["u"], dn, rv), False)]
''' + s[b:]
rep("    if cb is not None: clamp_to(cb, crnd)", "    if cb is not None: steer(*cb)")
a = s.index('if METHOD == "probe":\n    A = lambda'); b = s.index('json.dump(pr, open(f"{OUT}/e21_probe.json", "w")); print("DONE", f"{time.time()-t0:.0f}s")\n')
b += len('json.dump(pr, open(f"{OUT}/e21_probe.json", "w")); print("DONE", f"{time.time()-t0:.0f}s")\n')
s = s[:a] + r'''if METHOD == "probe":
    J = J.copy(); J["c"] = J.wrap.str.split(":").str[0]; J["w"] = J.wrap.str.split(":").str[1]
    def asr(c, fam): q = J[(J.c == c) & (J.w.isin(fam))]; return q
    out = dict(model=MODEL, layer=L, coverage=COV, directions=TB.to_dict("records"),
               by_wrapper={r["name"]: r["by_wrapper"] for r in ROWS}, none_fic=float(asr("none", FIC_TEST).yes.mean() * 100), none_bare=float(asr("none", ["bare"]).yes.mean() * 100), tests={})
    rng = np.random.default_rng(0)
    for r in SELECTED:
        n = r["name"]
        for kind, fam, a_, b_, sign in [("up", ["bare"], f"{n}_up", f"{n}_uprand", 1), ("down", FIC_TEST, f"{n}_downrand", f"{n}_down", 1)]:
            A_ = asr(a_, fam).set_index(["goal", "w"]).yes.astype(int); B_ = asr(b_, fam).set_index(["goal", "w"]).yes.astype(int)
            P = pd.concat([A_, B_], axis=1, keys=["a", "b"]).dropna(); d = 100 * (P.a - P.b).values
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
            out["tests"][f"{n}:{kind}"] = dict(asr_dir=float(asr(f"{n}_{kind}", fam).yes.mean() * 100), asr_rand=float(asr(f"{n}_{kind}rand", fam).yes.mean() * 100),
                                                selective=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                p_one_sided=float((np.sum(bs <= 0) + 1) / (len(bs) + 1)), n=int(len(d)))
    print("\n" + "=" * 80 + f"\nE23 | {MODEL} | layer {L}\n" + "=" * 80); print(json.dumps(out["coverage"]))
    for k, v in out["tests"].items(): print(f"   {k:>20}: dir {v['asr_dir']:5.1f} rand {v['asr_rand']:5.1f} selective {v['selective']:+5.1f} CI {v['ci95']} p {v['p_one_sided']:.4f}")
    json.dump(out, open(f"{OUT}/e23_summary.json", "w"), indent=1, default=str); print("DONE", f"{time.time()-t0:.0f}s")
''' + s[b:]
s = s.replace("e21_", "e23_").replace("\\nE21 |", "\\nE23 |")
open(OUTF, "w", encoding="utf-8").write(s); compile(s, "e23", "exec"); print("ok", len(s))
