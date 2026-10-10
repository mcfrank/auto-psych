"""Posterior summaries of each run winner's population-level scalars (own-run Figure 3 fits, cache only)."""
import importlib.util, json, sys
import arviz as az
from pyprojroot import here
sys.path.insert(0, str(here()))
S = str(here() / "data/results/human_experiment_sr-oct26/analysis/paper_update")
spec = importlib.util.spec_from_file_location("plot_human_fit_runs", f"{S}/plot_human_fit_runs.py")
ph = importlib.util.module_from_spec(spec); sys.modules["plot_human_fit_runs"] = ph; spec.loader.exec_module(ph)
from src.models.pymc_inference import clear_fit_cache, fit_model
out = {}
for run in (1, 2, 3):
    name = ph.BEST_AGENT_MODEL[run]
    responses = here() / f"data/results/human_experiment_sr-oct26/collected/run{run}/subjective_randomness/experiment3/model_loop/responses.csv"
    models_dir, seed, _ = ph.fit_source(name, run)
    f = fit_model(name, models_dir, responses, cache_dir=ph.ANALYSIS_ROOT / "mcmc_cache", random_seed=seed, **ph.SAMPLER)
    post = f.idata.posterior
    scalars = [v for v in post.data_vars if post[v].ndim == 2]
    s = az.summary(f.idata, var_names=scalars, kind="stats", hdi_prob=0.95)
    out[name] = {v: {k: round(float(s.loc[v, k]), 3) for k in s.columns} for v in s.index}
    print(name); print(s.round(3).to_string())
    del f; clear_fit_cache()
(here() / "data/results/human_experiment_sr-oct26/analysis/results/winner_parameters_runs123.json").write_text(json.dumps(out, indent=2))
