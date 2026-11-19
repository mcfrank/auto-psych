import pandas as pd, numpy as np, statsmodels.api as sm
d=pd.read_pickle('sr/strict2.pkl')
WIN=['ppdg','esp','mstc']; SEEDS=['lr','ms','fk','feo']
lg=lambda p: np.log(np.clip(p,1e-3,1-1e-3)/(1-np.clip(p,1e-3,1-1e-3)))
def ham(s):
    L=len(s); refs=['H'*L,'T'*L,('HT'*L)[:L],('TH'*L)[:L]]
    return min(sum(a!=b for a,b in zip(s,r)) for r in refs)
d['hamA']=d.A.map(ham); d['hamB']=d.B.map(ham)
d['nearA']=(d.hamA==1).astype(float); d['nearB']=(d.hamB==1).astype(float)
d['perfA']=(d.hamA==0).astype(float); d['perfB']=(d.hamB==0).astype(float)
d['motA']=(d.perA.isin([3,4])&(d.len>=6)).astype(float); d['motB']=(d.perB.isin([3,4])&(d.len>=6)).astype(float)
feats=['near','perf','mot','variety','lopsided','maxrun','pal','anti','startrun','endrun','trip']
X=pd.DataFrame({f:d['A_'+f if 'A_'+f in d else f+'A']-d['B_'+f if 'B_'+f in d else f+'B'] for f in feats})
r=lg(d.obs)-lg(d['stack'])
w=d.n*d['stack']*(1-d['stack'])
fit=sm.WLS(r,sm.add_constant(X),weights=w).fit()
print('residual of observed vs seed stack, regressed on A-B feature differences')
print(fit.summary2().tables[1][['Coef.','Std.Err.','t']].round(2))
for m in WIN:
    rm=lg(d[m])-lg(d['stack'])
    f2=sm.WLS(rm,sm.add_constant(X),weights=w).fit()
    print(m,'tracks:',f2.params.round(2).to_dict())
d.to_pickle('sr/strict3.pkl')
