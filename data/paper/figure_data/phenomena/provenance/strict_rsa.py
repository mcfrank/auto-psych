import pandas as pd, numpy as np, json
from scipy.optimize import minimize
df=pd.read_pickle('/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/rsa/trials.pkl')
F=json.load(open('data/rsa/live/finals_pool_preds_final.json')); B=json.load(open('data/rsa/live/bar_pool_preds.json'))
pool=F['pool']; off=np.r_[0,np.cumsum(pool['n_entries'])]
FIN=list(F['models']); SEEDS=list(B['models'])
P=np.exp(df[SEEDS].values)  # (trials, seeds): prob of chosen class
disp=df.ctx.values; ud=np.unique(disp); rng=np.random.default_rng(0); fd=dict(zip(ud,rng.integers(0,5,len(ud)))); fold=np.array([fd[c] for c in disp])
mix=np.zeros(len(df))
for f in range(5):
    tr=fold!=f
    nll=lambda z: -np.log(P[tr]@np.exp(z-z.max())/np.exp(z-z.max()).sum()+1e-12).sum()
    z=minimize(nll,np.zeros(len(SEEDS)),method='L-BFGS-B').x; w=np.exp(z-z.max()); w/=w.sum()
    mix[~tr]=P[~tr]@w
    if f==0: print('mixture weights (fold 0):',{s:round(v,2) for s,v in zip(SEEDS,w) if v>0.02})
df['mix']=np.log(mix+1e-12)
print('lpd:',{m:round(df[m].sum(),1) for m in FIN+['mix']}, ' best single seed:',df[SEEDS].sum().max().round(1), df[SEEDS].sum().idxmax())
# per-display
g=df.groupby('ctx'); d=pd.DataFrame({'n':g.size()})
for m in FIN+['mix']+SEEDS: d[m]=g[m].sum()
d['env']=d[SEEDS].max(axis=1)
for m in FIN: d['vs_mix_'+m[:8]]=d[m]-d['mix']; d['vs_env_'+m[:8]]=d[m]-d['env']
old=pd.read_pickle('/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/rsa/disp.pkl'); d=d.join(old[['n_obj','utt','n_match','interior','mf_range','dup']])
print({c:round(d[c].sum(),1) for c in d.columns if c.startswith('vs_')})
d['cat']=np.where(d.utt!=True,'no word',d.n_obj.astype(str)+' obj, word fits '+d.n_match.fillna(0).astype(int).astype(str))
print(d.groupby('cat')[[c for c in d.columns if c.startswith('vs_mix')]].sum().round(1))
print(d[d.utt==True].groupby('interior')[[c for c in d.columns if c.startswith('vs_')]].agg('sum').round(1))
d.to_pickle('/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/rsa/strict.pkl')
