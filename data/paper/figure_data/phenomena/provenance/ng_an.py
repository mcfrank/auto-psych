import pandas as pd, numpy as np, csv
from scipy.optimize import minimize
from scipy.special import expit, logit
S='/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/ng/'
C='data/results/number_game_dataset-nov26/collected/'
rows=[]
for r in (1,2,3): rows+=list(csv.DictReader(open(C+f'run{r}/number_game/experiment3/model_loop/responses.csv')))
R=pd.DataFrame(rows); R['rating']=R.rating.astype(float).astype(int)
it=R.groupby(['number_set','target']).rating.agg(['sum','size']).rename(columns={'sum':'k','size':'n'}).reset_index()
M={'sf':'span_fallback_rule_violation','nd':'neighbourhood_doubt_three_example_memory','ew':'exception_weighted_rule_rejection','bpi':'bigelow_piantadosi_independent','bpm':'bigelow_piantadosi_mixture','ten':'tenenbaum_rules_similarity'}
for k,m in M.items():
    for mode in ('full','cv'):
        x=pd.read_csv(S+f'{m}_{mode}.csv',dtype={'number_set':str,'target':str}).rename(columns={'p':f'{k}_{mode}'})
        it=it.merge(x,on=['number_set','target'])
it['obs']=it.k/it.n; k_,n_=it.k.values,it.n.values
def bll(p): p=np.clip(p,1e-6,1-1e-6); return k_*np.log(p)+(n_-k_)*np.log(1-p)
sat=bll(it.obs).sum(); base=bll(np.full(len(it),k_.sum()/n_.sum())).sum()
print(f'items {len(it)}, responses {n_.sum()}; saturated {sat:.0f}; constant yes-rate {base:.0f}')
sets=sorted(it.number_set.unique()); rng=np.random.default_rng(0); fold={s:i%5 for i,s in zip(range(len(sets)),rng.permutation(sets))}; it['fold']=it.number_set.map(fold)
def recal(col):
    x=logit(np.clip(it[col].values,1e-4,1-1e-4)); out=np.zeros(len(it))
    for f in range(5):
        tr=(it.fold!=f).values
        nll=lambda th: -np.sum(bll(expit(th[0]+th[1]*x))[tr])
        th=minimize(nll,[0,1]).x; out[~tr]=expit(th[0]+th[1]*x[~tr])
    return out
tab=[]
for k in M:
    rc=recal(k+'_cv')
    tab.append(dict(model=k, insample=bll(it[k+'_full']).sum(), cv=bll(it[k+'_cv']).sum(), cv_recalibrated=bll(rc).sum(),
                    r2_cv=np.corrcoef(it[k+'_cv'],it.obs)[0,1]**2, spearman=pd.Series(it[k+'_cv']).corr(it.obs,method='spearman')))
    it[k+'_rc']=rc
print(pd.DataFrame(tab).round(3).to_string(index=False))
it.to_pickle(S+'items.pkl')
