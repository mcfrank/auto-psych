"""MAP-fit a number-game model on the pooled responses (optionally grouped-CV by example set);
write per-item P(yes)."""
import sys, csv, time, numpy as np, pandas as pd, pymc as pm
from pathlib import Path
from src.models.model_loading import load_pymc_model
from src.models.data_binding import extract_observed, make_stim_data
S=Path('/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/ng')
C='data/results/number_game_dataset-nov26/collected/'
DIRS={'span_fallback_rule_violation':C+'run1/number_game/experiment3/cognitive_models',
      'neighbourhood_doubt_three_example_memory':C+'run2/number_game/experiment3/cognitive_models',
      'exception_weighted_rule_rejection':C+'run3/number_game/experiment3/cognitive_models'}
SEEDS='src/pipelines/outer_loop/projects/number_game/seed_models'
name=sys.argv[1]; mode=sys.argv[2] if len(sys.argv)>2 else 'full'
mdir=Path(sys.argv[3]) if len(sys.argv)>3 else Path(DIRS.get(name,SEEDS))
rows=[]
for r in (1,2,3):
    with open(C+f'run{r}/number_game/experiment3/model_loop/responses.csv') as f: rows+=list(csv.DictReader(f))
items=pd.DataFrame(rows)[['number_set','target']].drop_duplicates().reset_index(drop=True)
sets=sorted(items.number_set.unique()); rng=np.random.default_rng(0); fold_of={s:i%5 for i,s in zip(range(len(sets)), rng.permutation(sets))}
def write_csv(rs,p):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['number_set','target','trial_index','rating']); w.writeheader(); w.writerows(rs)
def fit_predict(train_rows, pred_items):
    p=S/f'_train_{name}_{mode}.csv'; write_csv(train_rows,p)
    model=load_pymc_model(name, mdir)
    obs=extract_observed(p, model)
    with model:
        pm.set_data(obs)
        t=time.time(); mp=pm.find_MAP(progressbar=False, maxeval=20000); 
    pr=[dict(number_set=a,target=b,trial_index=0,rating=0) for a,b in zip(pred_items.number_set,pred_items.target)]
    stim=make_stim_data(model, pr)
    with model:
        pm.set_data(stim)
        (py_,)=model.replace_rvs_by_values([model['p_yes']]); f=model.compile_fn(py_, inputs=model.value_vars, on_unused_input='ignore')
        pv=f({v.name: mp[v.name] for v in model.value_vars})
    return np.asarray(pv), mp, time.time()-t
if mode=='full':
    pv,mp,dt=fit_predict(rows, items); items['p']=pv
    print(name,'fit s',round(dt), {k:np.round(v,3).tolist() for k,v in mp.items() if np.size(v)<4 and not k.endswith('__')})
else:
    items['p']=np.nan
    for k in range(5):
        tr=[r for r in rows if fold_of[r['number_set']]!=k]; te=items.number_set.map(fold_of)==k
        pv,mp,dt=fit_predict(tr, items[te]); items.loc[te,'p']=pv; print(name,'fold',k,'s',round(dt),flush=True)
items.to_csv((Path(sys.argv[3]) if len(sys.argv)>3 else S)/f'{name}_{mode}.csv',index=False)
