import pandas as pd, numpy as np, sys, glob
from scipy.stats import binomtest
N=sys.argv[1]; L=lambda p: pd.read_csv(glob.glob(p)[0])
FIC=['story','roleplay','hypothetical','game']; PERM=['authorized','legal','operator']
def vec(df,s,w=None):
    d=df[df.set==s]; d=d[d.wrap.isin(w)] if w else d
    return d.sort_values(['wrap','goal']).yes.astype(int).values
def cmp(a,b,alt='two-sided'):
    n01=((a==1)&(b==0)).sum(); n10=((a==0)&(b==1)).sum(); p=binomtest(n01,n01+n10,alternative=alt).pvalue if n01+n10 else 1
    rng=np.random.default_rng(0); n=len(a); d=[a[i].mean()-b[i].mean() for i in (rng.integers(0,n,n) for _ in range(4000))]
    lo,hi=np.percentile(d,[2.5,97.5]); return f'{100*(a.mean()-b.mean()):+6.1f} [{100*lo:+.1f},{100*hi:+.1f}] p={p:.2g} (n01={n01}, n10={n10})'
SETS=[('fiction','jb',FIC),('permission','jb',PERM),('bare','jb',['bare']),('XSTest','xstest',None),('OR-Bench','orbench',None),('creative','creative',None)]
for m in ['qwen','gemma']:
    D={k:L(f'{N}/o21/{m}-{k.replace("_","-")}/e21_labels*.csv') for k in ['natural','bsct','act_paper']}
    print(f'\n{m.upper()}  (% ; diffs are paired, McNemar two-sided)')
    for name,s,w in SETS:
        v={k:vec(D[k],s,w) for k in D}
        print(f'  {name:10s} natural {100*v["natural"].mean():5.1f} | BSCT {100*v["bsct"].mean():5.1f} | ACT {100*v["act_paper"].mean():5.1f} | BSCT-ACT {cmp(v["bsct"],v["act_paper"])}')
    f={k:vec(D[k],'jb',FIC) for k in D}
    print(f'  PREREG TEST fiction: one-sided p(BSCT<ACT) = {cmp(f["bsct"],f["act_paper"],"less").split("p=")[1]}')
    for k in ['bsct','act_paper']:
        r=pd.read_json(glob.glob(f'{N}/o21/{m}-{k.replace("_","-")}/e21_result*.json')[0],typ='series'); print(f'  {k} MMLU {r.mmlu:.3f} degenerate {r.degenerate:.3f}')
    print(f'  natural MMLU', pd.read_json(glob.glob(f'{N}/o21/{m}-natural/e21_result*.json')[0],typ='series').mmlu)
    for k in ['bsct','act_paper']:
        sh=pd.read_csv(glob.glob(f'{N}/o21/{m}-{k.replace("_","-")}/e21_shift*.csv')[0]); print(f'  SHIFT {k}')
        print(sh.groupby('family')[['shift_norm','frac_in_gating','random_frac','cos_real','cos_illegal','cos_policy_forbids']].mean().round(3).to_string())
