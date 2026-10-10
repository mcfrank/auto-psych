"""Pairwise RMSE between the three runs' winners' population-level predictions.

Each winner as fitted on its own run's data (the Figure 3 fits, read from the
cache only), predicted on the loop's 512-pair novelty pool (lengths 4-8,
``novelty_pool_rows``), person-level models averaged over that run's
participants (``_pool_prediction``, as the novelty gate does), as
report_fits.py computed run 1's 0.092 / 0.029. Also each run's best seed
(local_representativeness, fitted on the same run) for scale.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys

import numpy as np
from pyprojroot import here

sys.path.insert(0, str(here()))
S = str(here() / "data/results/human_experiment_sr-oct26/analysis/paper_update")
spec = importlib.util.spec_from_file_location("plot_human_fit_runs", f"{S}/plot_human_fit_runs.py")
ph = importlib.util.module_from_spec(spec)
sys.modules["plot_human_fit_runs"] = ph
spec.loader.exec_module(ph)

from src.models.pymc_inference import clear_fit_cache, convergence_problems_of, fit_model  # noqa: E402
from src.pipelines.inner_loop.model_zoo import _participant_ids_in, _pool_prediction, novelty_pool_rows  # noqa: E402

pool = novelty_pool_rows()
preds = {}
for run in (1, 2, 3):
    responses = here() / f"data/results/human_experiment_sr-oct26/collected/run{run}/subjective_randomness/experiment3/model_loop/responses.csv"
    pids = _participant_ids_in(responses)
    for name in (ph.BEST_AGENT_MODEL[run], "local_representativeness"):
        models_dir, seed, _ = ph.fit_source(name, run)
        fitted = fit_model(name, models_dir, responses, cache_dir=ph.ANALYSIS_ROOT / "mcmc_cache", random_seed=seed, **ph.SAMPLER)
        if convergence_problems_of(fitted):
            raise ValueError(f"{name}@run{run} not converged")
        preds[f"run{run}:{name}"] = _pool_prediction(fitted, pool, model_name=name, participant_ids=pids)
        print(f"[pred] run {run} {name}: {np.isnan(preds[f'run{run}:{name}']).sum()} undefined of {len(pool)}", flush=True)
        del fitted
        clear_fit_cache()

out = {}
for a, b in itertools.combinations(preds, 2):
    ok = ~np.isnan(preds[a]) & ~np.isnan(preds[b])
    out[f"{a} vs {b}"] = float(np.sqrt(np.mean((preds[a][ok] - preds[b][ok]) ** 2)))
print(json.dumps(out, indent=2))
(here() / "data/results/human_experiment_sr-oct26/analysis/results/winner_pool_rmse_runs123.json").write_text(json.dumps(out, indent=2))
np.savez(f"{S}/winner_pool_predictions.npz", **{k.replace(":", "__"): v for k, v in preds.items()})
