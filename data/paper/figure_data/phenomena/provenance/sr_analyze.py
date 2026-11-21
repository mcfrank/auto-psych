import pandas as pd, numpy as np
S='/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/sr/'
r = pd.read_csv('data/results/human_experiment_sr-oct26/analysis/fits/pooled_runs123/responses.csv')
M = ['ppdg','esp','mstc','lr','ms','fk','feo']
for m in M:
    d = pd.read_csv(S+m+'.csv'); assert (d.obs.values==r.chose_left.values).all(), m
    r[m+'_loo']=d.loo.values; r[m+'_lppd']=d.lppd.values; r[m+'_p']=d.p_left.values
r['run']=r['run'] if 'run' in r else np.nan
print(r.columns.tolist()[:8], len(r), r.participant_id.nunique())
# canonical orientation
r['A']=np.where(r.sequence_a<r.sequence_b, r.sequence_a, r.sequence_b)
r['B']=np.where(r.sequence_a<r.sequence_b, r.sequence_b, r.sequence_a)
flip = r.sequence_a!=r.A
r['choseA']=np.where(flip, 1-r.chose_left, r.chose_left)
for m in M: r[m+'_pA']=np.where(flip, 1-r[m+'_p'], r[m+'_p'])
r['len']=r.A.str.len()
r.to_pickle(S+'merged.pkl')
print('pairs', r.groupby(['A','B']).ngroups, 'lengths', r.len.value_counts().sort_index().to_dict())
tot = {m: r[m+'_loo'].sum() for m in M}; print({k:round(v,1) for k,v in tot.items()})
for w in ['ppdg','esp','mstc']:
    r['d']=r[w+'_loo']-r['lr_loo']
    g=r.groupby(['A','B']).d.sum().sort_values()
    p=r.groupby('participant_id').d.sum().sort_values()
    print(f'\n== {w} vs LR: total {r.d.sum():.0f}; by pair: share of gain from top 10% pairs {g[g>0].nlargest(int(len(g)*.1)).sum()/r.d.sum():.2f}; pairs where winner worse {(g<0).sum()}/{len(g)} (sum {g[g<0].sum():.0f})')
    print(f'   by person: top 10% people give {p.nlargest(int(len(p)*.1)).sum()/r.d.sum():.2f}; people worse {(p<0).sum()}/{len(p)}')
