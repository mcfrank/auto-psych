import pandas as pd, numpy as np, statsmodels.api as sm
from scipy.optimize import minimize
d=pd.read_pickle('ng/items2.pkl').copy()
d['ex']=d.number_set.astype(str).str.split(',').apply(lambda s:[int(v) for v in s]); d['t']=d.target.astype(int)
lg=lambda p: np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
def ll(p,k,n): p=np.clip(p,1e-9,1-1e-9); return k*np.log(p)+(n-k)*np.log(1-p)
SEEDS=['bpi','bpm','ten']; WIN=['sf','nd','ew']
k=d.k.values; n=d.n.values; fold=d.fold.values
single=(d.n_ex==1).astype(float).values
X=np.column_stack([np.ones(len(d)),single]+[lg(d[s+'_cv'].values) for s in SEEDS]+[lg(d[s+'_cv'].values)*single for s in SEEDS])
stack=np.zeros(len(d))
for f in np.unique(fold):
    tr=fold!=f
    nll=lambda b: -ll(1/(1+np.exp(-np.clip(X[tr]@b,-30,30))),k[tr],n[tr]).sum()+0.01*(b**2).sum()
    b=minimize(nll,np.zeros(X.shape[1]),method='BFGS').x
    stack[~tr]=1/(1+np.exp(-X[~tr]@b))
d['stack']=stack
for m in WIN+SEEDS: d['ll_'+m]=ll(d[m+'_cv'],k,n)
for s in SEEDS: d['ll_'+s+'rc']=ll(d[s+'_rc'],k,n)
d['ll_stack']=ll(stack,k,n); d['ll_sat']=ll(d.obs,k,n)
print('item-level CV LL:',{m:round(d['ll_'+m].sum()) for m in ['sat','stack']+WIN+SEEDS+[s+'rc' for s in SEEDS]})
print('winner minus CV stack of the 3 seeds:',{w:round((d['ll_'+w]-d.ll_stack).sum()) for w in WIN})
# features
def F(r):
    ex,t=r.ex,r.t; o={}
    o['single_multiple']=float(len(ex)==1 and t%ex[0]==0 and t!=ex[0])
    o['single_factor']=float(len(ex)==1 and ex[0]%t==0 and t!=ex[0])
    o['digit_keep']=float(len(ex)>=2 and len({e%10 for e in ex})==1 and t%10==ex[0]%10)
    o['digit_break']=float(len(ex)>=2 and len({e%10 for e in ex})==1 and t%10!=ex[0]%10)
    par=len(ex)>=2 and len({e%2 for e in ex})==1
    o['parity_keep']=float(par and t%2==ex[0]%2)
    o['parity_break']=float(par and t%2!=ex[0]%2)
    g=np.gcd.reduce(ex); o['divisor_break']=float(len(ex)>=2 and g>=3 and t%g!=0)
    o['divisor_keep']=float(len(ex)>=2 and g>=3 and t%g==0)
    o['gap']=max(0,min(ex)-t,t-max(ex))/100
    o['last_digit_match']=float(len(ex)>=2 and t%10==ex[-1]%10)
    o['first_digit_match']=float(len(ex)>=2 and t%10==ex[0]%10)
    o['round_target']=float(t%10==0)
    o['is_example']=float(t in ex)
    return pd.Series(o)
Fx=d.apply(F,axis=1)
r=lg(d.obs)-lg(d['stack']); w=d.n*d['stack']*(1-d['stack'])
fit=sm.WLS(r,sm.add_constant(Fx),weights=w).fit()
out=pd.DataFrame({'resid_coef':fit.params,'t':fit.tvalues})
for m in WIN: out[m+'_tracks']=sm.WLS(lg(d[m+'_cv'])-lg(d['stack']),sm.add_constant(Fx),weights=w).fit().params
pd.set_option('display.width',250); print(out.round(2))
d=pd.concat([d,Fx],axis=1); d.to_pickle('ng/strict.pkl')
