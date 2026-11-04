import pandas as pd, numpy as np, sys, glob
from scipy.stats import binomtest
N,O=sys.argv[1],sys.argv[2]
L=lambda p: pd.read_csv(glob.glob(p)[0])
FIC=['story','roleplay','hypothetical','game']; PERM=['authorized','legal','operator']
def vec(df,s,w=None):
    d=df[df.set==s]; d=d[d.wrap.isin(w)] if w else d
    return d.sort_values(['wrap','goal']).yes.astype(int).values
def cmp(a,b):
    n01=((a==1)&(b==0)).sum(); n10=((a==0)&(b==1)).sum(); p=binomtest(n01,n01+n10).pvalue if n01+n10 else 1
    rng=np.random.default_rng(0); n=len(a); d=[a[i].mean()-b[i].mean() for i in (rng.integers(0,n,n) for _ in range(4000))]
    lo,hi=np.percentile(d,[2.5,97.5]); return f'{100*(a.mean()-b.mean()):+6.1f} [{100*lo:+.1f},{100*hi:+.1f}] p={p:.2g}'
SETS=[('fiction','jb',FIC),('permission','jb',PERM),('bare','jb',['bare']),('XSTest','xstest',None),('OR-Bench','orbench',None)]
M={'bsct':[L(f'{O}/bsct/e19_labels_bsct.csv'),L(f'{N}/o20/min_bsct_s1/e19_labels*.csv'),L(f'{N}/o20/min_bsct_s2/e19_labels*.csv')],
   'act':[L(f'{O}/act_paper/e19_labels_act_paper.csv'),L(f'{N}/o20/min_act_paper_s1/e19_labels*.csv'),L(f'{N}/o20/min_act_paper_s2/e19_labels*.csv')]}
print('MINISTRAL, 3 seeds: per-seed %, mean, and pooled paired test (BSCT - ACT, seed i vs seed i)')
for name,s,w in SETS:
    b=[100*vec(x,s,w).mean() for x in M['bsct']]; a=[100*vec(x,s,w).mean() for x in M['act']]
    pb=np.concatenate([vec(x,s,w) for x in M['bsct']]); pa=np.concatenate([vec(x,s,w) for x in M['act']])
    print(f'  {name:10s} BSCT {np.round(b,1)} mean {np.mean(b):5.1f}+-{np.std(b,ddof=1):.1f} | ACT {np.round(a,1)} mean {np.mean(a):5.1f}+-{np.std(a,ddof=1):.1f} | diff {cmp(pb,pa)}')
print('\nLLAMA-3.1-8B (seed 0)')
LL={k:L(f'{N}/o20/llama_{k}/e19_labels*.csv') for k in ['natural','bsct','act_paper']}
for name,s,w in SETS:
    nb=vec(LL['natural'],s,w); print(f'  {name:10s} natural {100*nb.mean():5.1f} | BSCT {100*vec(LL["bsct"],s,w).mean():5.1f} ({cmp(vec(LL["bsct"],s,w),nb)}) | ACT {100*vec(LL["act_paper"],s,w).mean():5.1f} ({cmp(vec(LL["act_paper"],s,w),nb)}) | BSCT-ACT {cmp(vec(LL["bsct"],s,w),vec(LL["act_paper"],s,w))}')
print(pd.DataFrame({k:LL[k][LL[k].set=='jb'].groupby('wrap').yes.mean().mul(100).round(1) for k in LL}).T.to_string())
for k in ['bsct','act_paper']:
    sh=pd.read_csv(glob.glob(f'{N}/o20/llama_{k}/e19_shift*.csv')[0]); print(f'\nLLAMA SHIFT {k}')
    print(sh.groupby('family')[['shift_norm','frac_in_gating','random_frac','cos_real','cos_illegal','cos_policy_forbids','cos_watched']].mean().round(3).to_string())
for sd in [1,2]:
    sh=pd.read_csv(glob.glob(f'{N}/o20/min_act_paper_s{sd}/e19_shift*.csv')[0]); print(f'\nMINISTRAL ACT SHIFT seed {sd}')
    print(sh.groupby('family')[['frac_in_gating','random_frac','cos_real','cos_illegal']].mean().round(3).to_string())
