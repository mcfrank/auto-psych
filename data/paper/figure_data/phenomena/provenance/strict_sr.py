import pandas as pd, numpy as np
exec(open('qual_sr.py').read().split("def row")[0])   # reuse orientation/feature code
SEEDS=['lr','ms','fk','feo']; WIN=['ppdg','esp','mstc']
d['env']=d[['ll_'+s for s in SEEDS]].max(axis=1)
d['envseed']=d[['ll_'+s for s in SEEDS]].idxmax(axis=1).str[3:]
for w in WIN: d['x_'+w]=d['ll_'+w]-d.env
print('excess over per-pair best seed (sum, sum of positives, #pairs >2 nats):')
for w in WIN: print(w, d['x_'+w].sum().round(0), d['x_'+w].clip(lower=0).sum().round(0), (d['x_'+w]>2).sum())
# which pairs: winners all beat envelope by > 2 nats
both=(d[['x_'+w for w in WIN]]>2).all(axis=1)
print('\npairs where all 3 winners beat every seed by >2 nats:',both.sum())
cols=['A','B','n','obs','ppdg','esp','mstc','lr','ms','fk','feo','envseed','x_ppdg','x_esp','x_mstc']
pd.set_option('display.width',250)
print(d[both].sort_values('x_mstc',ascending=False)[cols].round(2).to_string(index=False))
print('\nrun-1-only (x_ppdg>3, others <1):')
r1=(d.x_ppdg>3)&(d.x_esp<1)&(d.x_mstc<1)
print(d[r1].sort_values('x_ppdg',ascending=False)[cols].head(20).round(2).to_string(index=False))
d.to_pickle('sr/strict.pkl')
