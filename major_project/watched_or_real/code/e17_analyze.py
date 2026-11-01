import pandas as pd, numpy as np, glob, sys
from scipy.stats import binomtest
S=sys.argv[1]
M={}
for k,f in [('natural','natural'),('gated','gated_clamp'),('bsct','bsct'),('act','act')]:
    M[k]=pd.read_csv(f'{S}/out17/{k}/e17_labels_{f}.csv')
FIC=['story','roleplay','hypothetical','game']
def vec(df,s,col,wraps=None):
    d=df[df.set==s]
    if wraps is not None: d=d[d.wrap.isin(wraps)]
    return d.sort_values(['wrap','goal']).reset_index(drop=True)[col].astype(int).values
def mcn(a,b):
    n01=((a==1)&(b==0)).sum(); n10=((a==0)&(b==1)).sum()
    p=binomtest(n01,n01+n10).pvalue if n01+n10 else 1
    return n01,n10,p
def boot(a,b,B=5000):
    rng=np.random.default_rng(0); n=len(a); d=[]
    for _ in range(B):
        i=rng.integers(0,n,n); d.append(a[i].mean()-b[i].mean())
    return np.percentile(d,[2.5,97.5])
print(M['natural'].set.value_counts().to_dict())
rows=[('fiction ASR','jb','yes',FIC),('professional ASR','jb','yes',['professional']),('bare ASR','jb','yes',['bare']),
      ('XSTest answer','xstest','yes',None),('OR-Bench answer','orbench','yes',None),('creative answer','creative','yes',None)]
for name,s,col,w in rows:
    base=vec(M['natural'],s,col,w)
    print(f'\n{name} (n={len(base)}): natural {base.mean()*100:.1f}')
    for k in ['gated','bsct','act']:
        v=vec(M[k],s,col,w); lo,hi=boot(v,base); _,_,p=mcn(v,base)
        print(f'   {k:6s} {v.mean()*100:5.1f}  diff {100*(v.mean()-base.mean()):+6.1f} [{100*lo:+.1f},{100*hi:+.1f}] McNemar p={p:.2g}')
    a=vec(M['bsct'],s,col,w); b=vec(M['act'],s,col,w); lo,hi=boot(a,b); _,_,p=mcn(a,b)
    print(f'   BSCT-ACT diff {100*(a.mean()-b.mean()):+.1f} [{100*lo:+.1f},{100*hi:+.1f}] p={p:.2g}')
for k in M:
    d=M[k][M[k].set=='jb']; print(k, d.groupby('wrap').yes.mean().round(3).to_dict())
