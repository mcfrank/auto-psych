"""Table 1 (all runs pooled) and Table S1 (each winner on the two other runs).

Reads paper_fits.py's fits (cache only, at each model's recorded converged
seed) and writes, per dataset, the rows of report_fits.py's table: k
(population-level scalars, + person-level vectors), ELPD-LOO, ΔELPD and SE(Δ)
from az.compare, the stimulus-clustered SE(Δ), and per-stimulus RMSE / R² of
the in-sample posterior-mean predictions (person-level models: each person's
own fitted prediction), the stimulus being the unordered pair scored as the
proportion choosing the alphabetically first sequence (report_fits.py).
Also each model's LOO reliability and convergence.

Writes analysis/results/paper_table_<dataset>.csv.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from pyprojroot import here

sys.path.insert(0, str(here()))
S = str(here() / "data/results/human_experiment_sr-oct26/analysis/paper_update")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pf = load("paper_fits", f"{S}/paper_fits.py")
rf = load("report_fits", here() / "data/results/human_experiment_sr-oct26/analysis/report_fits.py")
frm = load("fit_run_models", here() / "data/results/human_experiment_sr-oct26/analysis/fit_run_models.py")

import arviz as az  # noqa: E402

from src.models.clustered_se import cluster_dse, stimulus_clusters  # noqa: E402
from src.models.data_binding import make_stim_data  # noqa: E402
from src.models.pymc_inference import (  # noqa: E402
    cached_fit_path,
    clear_fit_cache,
    convergence_problems_of,
    fit_fingerprint,
    fit_model,
    resolve_fit_settings,
)

WINNER_RUN = {v: k for k, v in pf.WINNERS.items()}


def table(dataset: str) -> pd.DataFrame:
    out = pf.dataset_dir(dataset)
    models_dir, responses = out / "models", out / "responses.csv"
    names = [*pf.SEED_MODELS, *(pf.WINNERS[r] for r in pf.DATASETS[dataset][1])]
    with responses.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    resp = pd.read_csv(responses)
    stim = rf.stimulus_frame(resp)
    n_people = resp.participant_id.nunique()
    loo, info = {}, {}
    for name in names:
        seed = json.loads((out / "fit_seeds" / f"{name}.json").read_text(encoding="utf-8"))["seed"]
        kwargs = {**pf.SAMPLER, "random_seed": seed}
        nc = cached_fit_path(pf.ANALYSIS_ROOT / "mcmc_cache", name,
                             fit_fingerprint(name, models_dir, responses, resolve_fit_settings(name, models_dir, kwargs)))
        if not nc.exists():
            raise FileNotFoundError(f"{dataset} {name}: no cached fit {nc}")
        fitted = fit_model(name, models_dir, responses, cache_dir=pf.ANALYSIS_ROOT / "mcmc_cache", **kwargs)
        problems = convergence_problems_of(fitted)
        if problems:
            raise ValueError(f"{dataset} {name}: not converged: {problems}")
        diag = fitted.loo_diagnostics()
        p = fitted.predict_p_left(make_stim_data(fitted.model, rows), max_draws=1000)
        rmse, r2 = rf.stimulus_fit(stim, np.asarray(p))
        _, n_pop, n_person = frm.n_params(fitted.model, n_people)
        loo[name] = diag.loo
        info[name] = {
            "type": f"Win run {WINNER_RUN[name]}" if name in WINNER_RUN else "Seed",
            "k_population": n_pop,
            "k_person_vectors": n_person,
            "rmse_stimulus": rmse,
            "r2_stimulus": r2,
            "loo_unreliable": diag.unreliable,
            "max_pareto_k": diag.max_pareto_k,
            "seed": seed,
        }
        print(f"[{dataset}] {name}: ELPD {float(diag.loo.elpd_loo):.1f} RMSE {rmse:.3f} R2 {r2:.3f}", flush=True)
        del fitted
        clear_fit_cache()
    cmp = az.compare(loo, ic="loo")
    best = cmp.index[0]
    clusters = stimulus_clusters(rows)
    recs = []
    for rank, (name, row) in enumerate(cmp.iterrows()):
        recs.append({
            "dataset": dataset, "rank": rank, "model": name, **info[name],
            "n_trials": len(rows), "n_people": n_people,
            "elpd_loo": float(row["elpd_loo"]), "elpd_diff": float(row["elpd_diff"]), "dse": float(row["dse"]),
            "dse_clustered": 0.0 if name == best else cluster_dse(np.asarray(loo[best].loo_i), np.asarray(loo[name].loo_i), clusters),
        })
    df = pd.DataFrame(recs)
    df.to_csv(pf.ANALYSIS_ROOT / f"results/paper_table_{dataset}.csv", index=False)
    return df


if __name__ == "__main__":
    for dataset in sys.argv[1:] or list(pf.DATASETS):
        print(table(dataset).round(3).to_string(index=False), flush=True)
