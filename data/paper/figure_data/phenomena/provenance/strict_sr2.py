import pandas as pd, numpy as np
from scipy.optimize import minimize
d=pd.read_pickle('sr/strict.pkl')
SEEDS=['lr','ms','fk','feo']; WIN=['ppdg','esp','mstc']
lg=lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
def ll(p,k,n): p=np.clip(p,1e-9,1-1e-9); return k*np.log(p)+(n-k)*np.log(1-p)
def period(s):
    for p in range(1,len(s)):
        if all(s[i]==s[i+p] for i in range(len(s)-p)): return p
    return len(s)
d['perA']=d.A.map(period); d['perB']=d.B.map(period)
# CV stacking of all 4 seeds (+ length interactions), folds by pair
rng=np.random.default_rng(0); fold=rng.integers(0,5,len(d))
L=np.log(d.len.values)
def design(cols): 
    X=np.column_stack([np.ones(len(d))]+[lg(d[c].values) for c in cols]+[lg(d[c].values)*L for c in cols]); return X
X=design(SEEDS); k=d.k.values; n=d.n.values
stack=np.zeros(len(d))
for f in range(5):
    tr=fold!=f
    nll=lambda b: -ll(1/(1+np.exp(-X[tr]@b)),k[tr],n[tr]).sum()+0.01*(b**2).sum()
    b=minimize(nll,np.zeros(X.shape[1]),method='BFGS').x
    stack[~tr]=1/(1+np.exp(-X[~tr]@b))
d['stack']=stack; d['ll_stack']=ll(stack,k,n)
d['ll_sat']=ll(d.obs,k,n)
print('stimulus-level LL: sat',d.ll_sat.sum().round(0),' stack(CV)',d.ll_stack.sum().round(0),{m:d['ll_'+m].sum().round(0) for m in SEEDS+WIN})
for w in WIN: d['s_'+w]=d['ll_'+w]-d.ll_stack
print('winner minus CV stack of all seeds:',{w:d['s_'+w].sum().round(0) for w in WIN})
def cat(name,m):
    x=d[m]; r=dict(pattern=name,pairs=int(m.sum()),trials=int(x.n.sum()))
    for w in WIN: r['vs_stack_'+w]=x['s_'+w].sum()
    best=max(SEEDS,key=lambda s:x['ll_'+s].sum()); r['best_seed']=best
    for w in WIN: r['vs_best_'+w]=(x['ll_'+w]-x['ll_'+best]).sum()
    return r
const=lambda s: s.map(period)==1
perA,perB=d.perA,d.perB
one_const=(perA==1)!=(perB==1)
alt=lambda p,s: (p==2)
one_alt=((perA==2)&(d.len>=4))!=((perB==2)&(d.len>=4))
mot=lambda p: p.isin([3,4])&(d.len>=6)
one_mot=mot(perA)!=mot(perB)
rest=~one_const&~one_alt&~one_mot
rows=[cat('one side constant (HHHH..)',one_const),cat('one side perfect alternation',one_alt&~one_const),
      cat('one side a repeated 3-4 motif (HTTTHTTT, THHTHHTH)',one_mot&~one_const&~one_alt),cat('everything else',rest)]
for l in sorted(d.len.unique()): rows.append(cat(f'length {l}',d.len==l))
pd.set_option('display.width',250); print(pd.DataFrame(rows).round(0).to_string(index=False))
d.to_pickle('sr/strict2.pkl')
# motif pairs: observed vs predictions, oriented so X = motif sequence
m=one_mot&~one_const&~one_alt
x=d[m].copy(); fl=x.perB.isin([3,4])&(x.len>=6)
for c in ['obs','stack']+SEEDS+WIN: x['q_'+c]=np.where(fl,1-x[c],x[c])
print('\nmotif sequence chosen as more random: weighted mean')
print({c:round(np.average(x['q_'+c],weights=x.n),3) for c in ['obs','stack']+SEEDS+WIN})
x=d[one_alt&~one_const].copy(); fl=(x.perB==2)
for c in ['obs','stack']+SEEDS+WIN: x['q_'+c]=np.where(fl,1-x[c],x[c])
x['other_sw']=np.where(fl,x.A.map(lambda s:sum(a!=b for a,b in zip(s,s[1:]))),x.B.map(lambda s:sum(a!=b for a,b in zip(s,s[1:]))))/(x.len-1)
x['bin']=pd.cut(x.other_sw,[-.01,.3,.6,.8,.99])
print('\nperfect alternation chosen as more random, by the other sequence switch rate')
print(x.groupby('bin',observed=True).apply(lambda g: pd.Series({'pairs':len(g),**{c:np.average(g['q_'+c],weights=g.n) for c in ['obs','stack']+SEEDS+WIN}}),include_groups=False).round(2).to_string())
