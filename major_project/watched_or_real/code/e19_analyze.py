import pandas as pd, numpy as np, sys
from scipy.stats import binomtest
S=sys.argv[1]; M={k:pd.read_csv(f'{S}/o19/{k}/e19_labels_{k}.csv') for k in ['natural','bsct','bsct_sub','act_paper']}
FIC=['story','roleplay','hypothetical','game']; PERM=['authorized','legal','operator']
def vec(df,s,w=None):
    d=df[df.set==s]
    if w: d=d[d.wrap.isin(w)]
    return d.sort_values(['wrap','goal']).yes.astype(int).values
def cmp(a,b):
    n01=((a==1)&(b==0)).sum(); n10=((a==0)&(b==1)).sum(); p=binomtest(n01,n01+n10).pvalue if n01+n10 else 1
    rng=np.random.default_rng(0); n=len(a); d=[a[i].mean()-b[i].mean() for i in (rng.integers(0,n,n) for _ in range(4000))]
    lo,hi=np.percentile(d,[2.5,97.5]); return f'{100*(a.mean()-b.mean()):+6.1f} [{100*lo:+.1f},{100*hi:+.1f}] p={p:.2g}'
for name,s,w in [('fiction ASR','jb',FIC),('permission ASR','jb',PERM),('professional','jb',['professional']),('bare','jb',['bare']),('XSTest ans','xstest',None),('OR-Bench ans','orbench',None),('creative ans','creative',None)]:
    b=vec(M['natural'],s,w); print(f'\n{name} n={len(b)} natural {100*b.mean():.1f}')
    for k in ['bsct','bsct_sub','act_paper']:
        v=vec(M[k],s,w); print(f'   {k:9s} {100*v.mean():5.1f}  vs natural {cmp(v,b)}')
    print('   bsct - act_paper   ', cmp(vec(M['bsct'],s,w),vec(M['act_paper'],s,w)))
    print('   bsct_sub - act_paper', cmp(vec(M['bsct_sub'],s,w),vec(M['act_paper'],s,w)))
print()
print(pd.DataFrame({k:M[k][M[k].set=='jb'].groupby('wrap').yes.mean().mul(100).round(1) for k in M}))
print('\nCROSSED (natural, untrained):'); print(pd.read_csv(f'{S}/o19/natural/e19_crossed_natural.csv',index_col=0).round(2).to_string())
for k in ['bsct','bsct_sub','act_paper']:
    sh=pd.read_csv(f'{S}/o19/{k}/e19_shift_{k}.csv'); print(f'\nSHIFT {k}')
    print(sh.groupby('family')[['shift_norm','frac_in_gating','random_frac','cos_real','cos_policy_forbids','cos_illegal','cos_unauthorized','cos_malicious','cos_watched']].mean().round(3).to_string())
