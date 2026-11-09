# Post-review re-analysis of the E19-E24 defence comparisons (exploratory; written after the 2026-11-09 mock review).
# - Goal-clustered sign-flip permutation test: unit = JBB goal, statistic = mean over wrappers (and seeds) of the paired diff.
# - Seed-level view where seeds exist (Ministral E19/E20): per-seed estimates and a seed-level t-test.
# - Empirical noise floor: largest same-method difference between seeds.
# - Benjamini-Hochberg FDR across all goal-clustered tests.
# usage: python reanalysis_clustered.py  (paths relative to the repo's results/ folder)
import os, json, numpy as np, pandas as pd
from scipy.stats import ttest_1samp
R = os.path.join(os.path.dirname(__file__), "..", "results")
FIC = ["story", "roleplay", "hypothetical", "game"]
def L(p): return pd.read_csv(os.path.join(R, p))
RUNS = {
 "min": {"natural": ["E19_which_beliefs_to_protect/e19_labels_natural.csv"],
         "bsct": ["E19_which_beliefs_to_protect/e19_labels_bsct.csv", "E20_llama_replication_and_seeds/min_bsct_s1__labels_bsct.csv", "E20_llama_replication_and_seeds/min_bsct_s2__labels_bsct.csv"],
         "act_paper": ["E19_which_beliefs_to_protect/e19_labels_act_paper.csv", "E20_llama_replication_and_seeds/min_act_paper_s1__labels_act_paper.csv", "E20_llama_replication_and_seeds/min_act_paper_s2__labels_act_paper.csv"],
         "dir_pc0": ["E24_protect_the_used_belief/min-bsct-dir-pc0__e24_labels_bsct_dir.csv"],
         "combo_pc0": ["E24_protect_the_used_belief/min-bsct-combo-pc0__e24_labels_bsct_combo.csv"]},
 "llama": {"natural": ["E20_llama_replication_and_seeds/llama_natural__labels_natural.csv"],
           "bsct": ["E20_llama_replication_and_seeds/llama_bsct__labels_bsct.csv"],
           "act_paper": ["E20_llama_replication_and_seeds/llama_act_paper__labels_act_paper.csv"],
           "dir_pc2": ["E24_protect_the_used_belief/llama-bsct-dir-pc2__e24_labels_bsct_dir.csv"],
           "combo_pc2": ["E24_protect_the_used_belief/llama-bsct-combo-pc2__e24_labels_bsct_combo.csv"]},
 "qwen": {k: [f"E21_probe_predicts_defence/qwen_{k}__e21_labels_{k}.csv"] for k in ["natural", "bsct", "act_paper"]},
 "gemma": {k: [f"E21_probe_predicts_defence/gemma_{k}__e21_labels_{k}.csv"] for k in ["natural", "bsct", "act_paper"]},
}
def goal_matrix(df, goals=None):
    d = df[(df.set == "jb") & df.wrap.isin(FIC)]
    if goals is not None: d = d[d.goal.isin(goals)]
    return d.groupby("goal").yes.mean()                      # per-goal fiction ASR
def per_goal(m, name, goals=None):                           # seeds x goals
    return pd.concat([goal_matrix(L(p), goals) for p in RUNS[m][name]], axis=1).sort_index()
def signflip(d, n=20000, seed=0):
    rng = np.random.default_rng(seed); obs = d.mean()
    null = (rng.choice([-1, 1], (n, len(d))) * d).mean(1)
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))
def orbench(m, name):
    return [100 * L(p).query("set == 'orbench'").yes.mean() for p in RUNS[m][name]]
rows = []
PAIRS = [("bsct", "act_paper"), ("bsct", "natural"), ("act_paper", "natural"), ("dir_pc0", "bsct"), ("combo_pc0", "bsct"),
         ("dir_pc2", "bsct"), ("dir_pc2", "act_paper"), ("combo_pc2", "bsct")]
for m in RUNS:
    for a, b in PAIRS:
        if a not in RUNS[m] or b not in RUNS[m]: continue
        goals = list(range(50, 100)) if a.startswith(("dir_", "combo_")) else None
        A, B = per_goal(m, a, goals), per_goal(m, b, goals)
        d = 100 * (A.mean(1) - B.mean(1)).values
        r = dict(model=m, comparison=f"{a} - {b}", goals="50-99" if goals else "0-99", n_goals=len(d),
                 seeds=f"{A.shape[1]}x{B.shape[1]}", est=round(float(d.mean()), 2), p_goal_clustered=signflip(d),
                 orbench_a=round(float(np.mean(orbench(m, a))), 1), orbench_b=round(float(np.mean(orbench(m, b))), 1))
        if A.shape[1] == B.shape[1] > 1:                    # paired by seed index
            sd = [100 * (A.iloc[:, s] - B.iloc[:, s]).mean() for s in range(A.shape[1])]
            r["per_seed"] = [round(x, 2) for x in sd]; r["p_seed_level_t"] = float(ttest_1samp(sd, 0).pvalue)
        rows.append(r)
T = pd.DataFrame(rows)
p = T.p_goal_clustered.values; o = np.argsort(p); q = np.empty_like(p); k = len(p)
q[o] = np.minimum.accumulate((p[o] * k / np.arange(1, k + 1))[::-1])[::-1]; T["q_bh"] = np.minimum(q, 1)
floor = {}
for name in ["bsct", "act_paper"]:
    S = per_goal("min", name); fs = [100 * S.iloc[:, i].mean() for i in range(S.shape[1])]
    floor[name] = dict(per_seed_fiction=[round(x, 2) for x in fs], max_same_method_gap=round(max(fs) - min(fs), 2))
pd.set_option("display.width", 220)
print(T.to_string(index=False)); print("\nSame-method seed spread (Ministral fiction ASR, all goals):", json.dumps(floor))
out = os.path.join(R, "REANALYSIS_clustered"); os.makedirs(out, exist_ok=True)
T.to_csv(os.path.join(out, "clustered_tests.csv"), index=False); json.dump(floor, open(os.path.join(out, "seed_floor.json"), "w"), indent=1)
