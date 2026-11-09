# E26 preregistered analysis (results/E26_invariance_breadth/PREDICTIONS.md).
# usage: analyze.py <out_json> <llama_results_dir> <min_results_dir> <llama_pre_scores.json> <min_pre_scores.json>
# Each results dir holds the e26_result_*.json files of one model (any subfolders).
import sys, json, glob, numpy as np, pandas as pd
from scipy import stats
out_json, LD, MD, LP, MP = sys.argv[1:6]
def load(d):
    rows = [json.load(open(f)) for f in glob.glob(f"{d}/**/e26_result_*.json", recursive=True)]
    T = pd.DataFrame(rows)
    T["OOF"] = 100 * T[["roleplay_asr", "prefill_asr", "rspi_asr"]].mean(axis=1); T["IN"] = 100 * T.infam_asr
    T["ANS"] = 100 * T[["xstest_answer", "orbench_answer", "fic_borderline_answer"]].mean(axis=1)
    T["cell"] = T.arm + "-" + T.layers + "-" + T.pos + "-l" + T.lam.map(lambda x: f"{x:g}")
    return T
M = {"llama": load(LD), "min": load(MD)}; PRE = {"llama": json.load(open(LP)), "min": json.load(open(MP))}
def seeds(T, arm, col, layers="all", pos="span", lam=1.0):
    q = T[(T.arm == arm) & (T.lam == lam)] if arm in ("act", "bct") else T[(T.arm == arm) & (T.layers == layers) & (T.pos == pos) & (T.lam == lam)]
    return q.drop_duplicates("seed")[col].values.astype(float)
def welch(a, b):                      # mean(a) - mean(b), one-sided p for > 0, and the standard error / df
    d = a.mean() - b.mean(); va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b); se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else 1.0
    t = d / se if se > 0 else np.inf * np.sign(d); return float(d), float(stats.t.sf(t, df)) if se > 0 else float(d <= 0), float(se), float(df)
def partial_spearman(x, y, z, n=10000):
    rx, ry, rz = [stats.rankdata(v) for v in (x, y, z)]
    def pc(rx_):
        A = np.c_[np.ones(len(rz)), rz]; ex = rx_ - A @ np.linalg.lstsq(A, rx_, rcond=None)[0]; ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        return float(np.corrcoef(ex, ey)[0, 1])
    obs = pc(rx); rng = np.random.default_rng(0); null = np.array([pc(rng.permutation(rx)) for _ in range(n)])
    return obs, float((np.sum(null <= obs) + 1) / (n + 1))
H, P, R = {}, {}, {}
for m, T in M.items():
    nat = T[T.arm == "natural"].iloc[0]; testable = nat.OOF >= 10; R[m] = dict(natural_OOF=float(nat.OOF), testable=bool(testable))
    tr = T[T.method.isin(["inv", "act_paper", "bct"])]
    # H1 manipulation
    inv = tr[(tr.method == "inv") & (tr.lam > 0)]
    h1a = bool((inv.harm_last <= 0.5 * inv.harm_first).all())
    h1b = {a: float(tr[(tr.arm == a) & (tr.layers == "all") & (tr.pos == "span") & (tr.lam == 1)].push_removed_infam.mean()) for a in ["refusal", "pcs8", "full"]}
    H[f"H1_{m}"] = dict(held=bool(h1a and all(v > 0 for v in h1b.values())), loss_halved_all=h1a, push_removed=h1b, n=int(len(inv)))
    # H2 push law
    r, p = partial_spearman(tr.push_removed_infam.values, tr.OOF.values, tr.shift_removed_infam.values)
    P[f"H2_{m}"] = p; R[m]["H2"] = dict(partial_rho=r, p=p, n_adapters=int(len(tr)), point_ok=bool(r < 0))
    # H3 readout non-inferiority (in family): H0 diff >= 4
    a, b = seeds(T, "refusal", "IN"), seeds(T, "act", "IN"); d, _, se, df = welch(a, b)
    p3 = float(stats.t.cdf((d - 4) / se, df)) if se > 0 else float(d >= 4); P[f"H3_{m}"] = p3
    R[m]["H3"] = dict(diff=d, upper95=float(d + stats.t.ppf(0.95, df) * se) if se > 0 else d, p=p3, n=(len(a), len(b)), point_ok=True)
    for hid, (x, y) in {"H4": ("rand8", "pcs8"), "H5": ("real", "refusal")}.items():
        a, b = seeds(T, x, "OOF"), seeds(T, y, "OOF"); d, p_, se, df = welch(a, b); P[f"{hid}_{m}"] = p_
        R[m][hid] = dict(diff=d, p=p_, n=(len(a), len(b)), point_ok=bool(d >= 4), seeds_ok=bool(len(a) >= 2 and len(b) >= 2))
    # H6 / H6b
    pc8, fu = seeds(T, "pcs8", "OOF").mean(), seeds(T, "full", "OOF").mean()
    R[m]["R_pcs8"] = float((nat.OOF - pc8) / (nat.OOF - fu)) if nat.OOF != fu else np.nan
    al = T[(T.arm == "pcs8") & (T.layers == "all") & (T.pos == "last") & (T.lam == 1)].OOF.mean()
    R[m]["span_benefit"] = float(al - pc8); R[m]["kstar_win"] = PRE[m]["kstar_win"]; R[m]["full_win_frac_left"] = PRE[m]["full_win_frac_left"]
# BH over the 8 p-values
keys = sorted(P); pv = np.array([P[k] for k in keys]); o = np.argsort(pv); adj = np.empty_like(pv); n = len(pv)
adj[o] = np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1]; ADJ = dict(zip(keys, np.minimum(adj, 1).tolist()))
for hid in ["H2", "H3", "H4", "H5"]:
    per = {}
    for m in M:
        x = R[m][hid]; ok = x["point_ok"] and ADJ[f"{hid}_{m}"] < 0.05 and x.get("seeds_ok", True)
        if hid != "H2" and hid != "H3" and not R[m]["testable"]: ok = None
        if hid == "H2" and not R[m]["testable"]: ok = None
        per[m] = dict(**x, p_bh=ADJ[f"{hid}_{m}"], held=ok)
    H[hid] = dict(held=(None if any(v["held"] is None for v in per.values()) else bool(all(v["held"] for v in per.values()))), **per)
km = {m: R[m]["kstar_win"] for m in M}; rp = {m: R[m]["R_pcs8"] for m in M}
if km["llama"] != km["min"]:
    big, small = (("llama", "min") if km["llama"] > km["min"] else ("min", "llama"))
    h6a = bool(rp[big] < rp[small])
else: h6a = None
h6b = all((rp[m] >= 0.8) == (km[m] <= 8) for m in M)
H["H6"] = dict(held=(None if h6a is None else bool(h6a and h6b)), kstar_win=km, R_pcs8=rp, ordering_ok=h6a, threshold_ok=bool(h6b))
fw = {m: R[m]["full_win_frac_left"] for m in M}; sb = {m: R[m]["span_benefit"] for m in M}
bigp = max(fw, key=fw.get); H["H6b"] = dict(held=bool(sb[bigp] > sb[min(fw, key=fw.get)]), full_win_frac_left=fw, span_benefit=sb)
def tab(T):
    g = T.groupby("cell")[["IN", "OOF", "ANS", "push_removed_infam", "shift_removed_infam"]].agg(["mean", "count"]).round(2); g.columns = ["_".join(c) for c in g.columns]
    return g.reset_index().to_dict("records")
TAB = {m: tab(T) for m, T in M.items()}
json.dump(dict(hypotheses=H, details=R, bh=ADJ, tables=TAB), open(out_json, "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
print("PREREGISTERED:")
for k, v in H.items(): print(" ", k, {True: "HELD", False: "FAILED", None: "NOT TESTABLE"}[v["held"]], {a: b for a, b in v.items() if a != "held"})
