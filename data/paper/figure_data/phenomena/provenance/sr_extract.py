"""Per-trial PSIS-LOO and posterior-mean p_left for the pooled SR fits."""
import sys, h5py, numpy as np, pandas as pd, arviz as az, xarray as xr
from pathlib import Path
CACHE = Path('/Users/bp3902/Documents/auto-psych/data/results/human_experiment_sr-oct26/analysis/mcmc_cache')
OUT = Path('/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/sr')
fits = dict(
  ppdg='personal_pattern_detection_gain.9b305cb3880c2adb.nc',
  esp='edge_streak_primacy_recency_pattern_rule.045ef3ee541312e7.nc',
  esp_b='edge_streak_primacy_recency_pattern_rule.d00ee197748be577.nc',
  mstc='mild_symmetric_trick_coin_streak_weary.71cd754fb25a2ea8.nc',
  lr='local_representativeness.a0868d65d4a38b8c.nc',
  ms='motif_stack.26e8e23e038932be.nc',
  fk='falk_konold_dp.2ab68d2ab9761866.nc',
  feo='finite_experience_occurrence.bdd0424ee3f0d300.nc',
)
for key in sys.argv[1:]:
    f = CACHE / fits[key]
    with h5py.File(f, 'r') as h:
        ll = h['log_likelihood']['response'][...]          # chain, draw, n
        pl = h['posterior']['p_left'][...]
        obs = h['observed_data']['response'][...]
        sd = h['sample_stats']
        div = int(np.sum(sd['diverging'][...])) if 'diverging' in sd else -1
    idata = az.from_dict(posterior={'x': np.zeros(ll.shape[:2])}, log_likelihood={'response': ll})
    loo = az.loo(idata, pointwise=True)
    lppd = np.log(np.mean(np.exp(ll), axis=(0, 1)))
    df = pd.DataFrame(dict(loo=loo.loo_i.values, pareto_k=loo.pareto_k.values, lppd=lppd,
                           p_left=pl.mean(axis=(0, 1)), obs=obs))
    df.to_csv(OUT / f'{key}.csv', index=False)
    print(key, 'elpd_loo', round(loo.elpd_loo, 1), 'p_loo', round(loo.p_loo, 1), 'div', div, 'maxk', df.pareto_k.max().round(2))
