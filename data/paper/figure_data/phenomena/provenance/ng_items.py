import pandas as pd, numpy as np
S='/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/ng/'
it=pd.read_pickle(S+'items.pkl'); k_,n_=it.k.values,it.n.values
def bll(p): p=np.clip(p,1e-6,1-1e-6); return k_*np.log(p)+(n_-k_)*np.log(1-p)
primes={n for n in range(2,101) if all(n%a for a in range(2,int(n**.5)+1))}
def rule_of(ex):
    tests=[('ends in same digit',lambda e: len({x%10 for x in e})==1),('powers of 2',lambda e: all(x in (1,2,4,8,16,32,64) for x in e)),
           ('squares',lambda e: all(int(x**.5)**2==x for x in e)),('primes',lambda e: all(x in primes for x in e)),
           ('multiples of 10',lambda e: all(x%10==0 for x in e)),('multiples of 5',lambda e: all(x%5==0 for x in e)),
           ('common multiple ≥3',lambda e: any(all(x%m==0 for x in e) for m in range(3,13))),('all even',lambda e: all(x%2==0 for x in e)),('all odd',lambda e: all(x%2 for x in e))]
    for nm,t in tests:
        if t(ex): return nm
    return 'no rule'
ex=it.number_set.str.split(',').map(lambda v:[int(x) for x in v]); t=it.target.astype(int)
it['n_ex']=ex.map(len); it['rule']=[rule_of(e) if len(e)>1 else 'single example' for e in ex]
it['in_range']=[min(e)<=x<=max(e) for e,x in zip(ex,t)]
it['same_parity']=[all(y%2==x%2 for y in e) for e,x in zip(ex,t)]
it['is_example']=[x in e for e,x in zip(ex,t)]
for w in ['nd','ew','sf']:
    it['g_'+w]=bll(it[w+'_cv'])-bll(it['bpi_rc'])
it['g_raw_nd']=bll(it['nd_cv'])-bll(it['bpi_cv'])
pd.set_option('display.width',200)
agg=dict(items=('k','size'),resp=('n','sum'),nd_vs_raw_seed=('g_raw_nd','sum'),nd=('g_nd','sum'),ew=('g_ew','sum'),sf=('g_sf','sum'))
for c in ['n_ex','rule','in_range','is_example']:
    print(it.groupby(c).agg(**agg).round(0).to_string(),'\n')
print('share of nd gain (vs recalibrated seed) from top 10% items', round(it.g_nd.nlargest(214).sum()/it.g_nd.sum(),2), '; items nd worse:', (it.g_nd<0).sum())
cols=['number_set','target','n','obs','bpi_cv','bpi_rc','nd_cv','ew_cv','sf_cv','g_nd']
print(it.sort_values('g_nd',ascending=False)[cols].head(20).round(2).to_string(index=False))
it.to_pickle(S+'items2.pkl')
