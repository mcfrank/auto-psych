"""Fit the seed models and the loop's winners on each experiment's data (MCMC).

The compute half of the sr-oct26 analysis, the counterpart of the September
2026 paper analysis (``plot_human_fit_comparison.py`` and
``analysis/behavioral/fit_mega_models.py``, removed in 8d29ad1 / ce3d200).
Those could read the seeds' ELPD from ``model_posterior.json``; here the run's
starting models are prunable and run 1 pruned all four in experiment 1, so the
seeds are refit beside each experiment's winner(s) on that experiment's
cumulative ``model_loop/responses.csv``, at the live run's sampler settings.

Per experiment, writes to ``<out_dir>/experiment<N>/``:
  compare.json          ``compare_table`` over every model fitted there
                        (ELPD-LOO, elpd_diff, dse, dse_clustered, reliability)
  predictions.csv       per-trial posterior-mean p_left of every model
  n_params.csv          free random variables and their total scalar size
and for the final experiment ``pool_predictions.csv``: each model's
population-level p_left on the loop's novelty pool (person-level models
marginalised over the training participants, as the novelty gate does).

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/fit_run_models.py

On macOS 15+ with Apple clang 21, PyTensor's ``-ld64`` linker flag no longer
exists: point ``PYTENSOR_FLAGS=cxx=<dir>/clang++`` at a wrapper named clang++
that drops that one argument and runs /usr/bin/clang++.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

# The pipeline sizes its concurrent fits by the scheduler's affinity mask
# (``allocated_cpus``), which only Linux has; on a laptop, every CPU.
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid: set(range(os.cpu_count()))

RUN_ROOT = here() / "data/results/human_experiment_sr-oct26/collected/run1/subjective_randomness"
ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"

SEEDS = ("falk_konold_dp", "finite_experience_occurrence", "local_representativeness", "motif_stack")

# (model, experiment whose model_loop/models/ holds its source). Each
# experiment's data are fit with the seeds and the models listed for it.
MODELS_BY_EXPERIMENT = {
    1: [("individual_streak_aversion_lapse", 1)],
    2: [("most_lopsided_stretch_aversion", 2)],
    3: [
        # every experiment's winner, and the four models tied at the end
        ("individual_streak_aversion_lapse", 1),
        ("most_lopsided_stretch_aversion", 2),
        ("personal_pattern_detection_gain", 3),
        ("capacity_blur_pattern_gain", 3),
        ("returns_to_balance_expectation", 3),
        ("sensitivity_linked_consistency", 3),
    ],
}


@dataclass
class Args:
    """Refit the seeds and the run's winners on each experiment's data."""

    experiments: tuple[int, ...] = (1, 2, 3)
    out_dir: Path = ANALYSIS_ROOT / "fits"
    cache_dir: Path = ANALYSIS_ROOT / "mcmc_cache"
    # The live run's settings (full_run_sr-oct26.yaml).
    draws: int = 3000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.8
    random_seed: int = 20261003
    seed_overrides: dict[str, int] = field(default_factory=dict)
    """Sampler seeds for a fit whose chains stuck with ``random_seed``, keyed
    ``<model>@experiment<N>`` (or ``<model>`` for every experiment); found by
    find_converged_seed.py and recorded in compare.json."""
    predict_max_draws: int = 1000


def assemble_models_dir(experiment: int, work: Path) -> list[str]:
    """Copy the seeds and the experiment's models into ``work`` with a manifest."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    sources = [(s, RUN_ROOT / "experiment1/cognitive_models_input" / f"{s}.py") for s in SEEDS]
    sources += [
        (name, RUN_ROOT / f"experiment{src}/model_loop/models/{name}.py")
        for name, src in MODELS_BY_EXPERIMENT[experiment]
    ]
    lines = ["models:"]
    for name, path in sources:
        if not path.exists():
            raise FileNotFoundError(f"model source missing: {path}")
        shutil.copyfile(path, work / f"{name}.py")
        lines.append(f"  - name: {name}")
    (work / "models_manifest.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [name for name, _ in sources]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def n_params(model, n_participants: int) -> tuple[int, int, int]:
    """(free random variables, population-level scalars, person-level vectors).

    A free variable with at least one entry per participant is person-level
    (models size those by a ``MAX_PARTICIPANTS`` constant, most of it unused).
    """
    point = model.initial_point()
    sizes = [int(np.prod(point[model.rvs_to_values[rv].name].shape)) for rv in model.free_RVs]
    population = sum(n for n in sizes if n < n_participants)
    person = sum(1 for n in sizes if n >= n_participants)
    return len(sizes), population, person


def main(args: Args) -> None:
    import arviz as az

    from src.models.clustered_se import cluster_dse, stimulus_clusters
    from src.models.data_binding import make_stim_data
    from src.models.pymc_inference import (
        clear_fit_cache,
        convergence_problems_of,
        fit_model,
        fit_models_cached,
    )
    from src.pipelines.inner_loop.model_zoo import (
        _participant_ids_in,
        _pool_prediction,
        novelty_pool_rows,
    )

    def fit_kwargs(name: str) -> dict:
        return dict(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=args.seed_overrides.get(
                f"{name}@experiment{experiment}", args.seed_overrides.get(name, args.random_seed)
            ),
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for experiment in args.experiments:
        out = args.out_dir / f"experiment{experiment}"
        out.mkdir(parents=True, exist_ok=True)
        models_dir = out / "models"
        names = assemble_models_dir(experiment, models_dir)
        responses = RUN_ROOT / f"experiment{experiment}/model_loop/responses.csv"
        print(f"[fit] experiment {experiment}: {names} on {responses}", flush=True)

        # Sample every model into the cache (models sharing settings together),
        # then load them one at a time: a fit at 7,680 trials is ~1.5 GB.
        by_seed: dict[int, list[str]] = {}
        for name in names:
            by_seed.setdefault(fit_kwargs(name)["random_seed"], []).append(name)
        for group in by_seed.values():
            fit_models_cached(group, models_dir, responses, cache_dir=args.cache_dir, **fit_kwargs(group[0]))
            clear_fit_cache()

        rows = read_rows(responses)
        n_participants = len({row["participant_id"] for row in rows})
        pool = novelty_pool_rows() if experiment == max(MODELS_BY_EXPERIMENT) else None
        pids = _participant_ids_in(responses)
        loo, info, preds, n_par, pool_preds = {}, {}, [], [], []
        for name in names:
            fitted = fit_model(name, models_dir, responses, cache_dir=args.cache_dir, **fit_kwargs(name))
            diag = fitted.loo_diagnostics()
            loo[name] = diag.loo
            problems = convergence_problems_of(fitted)
            info[name] = {
                "loo_unreliable": diag.unreliable,
                "n_bad_k": diag.n_bad_k,
                "frac_bad_k": diag.frac_bad_k,
                "max_pareto_k": diag.max_pareto_k,
                "convergence_problems": problems,
                "not_converged": bool(problems),
                "random_seed": fit_kwargs(name)["random_seed"],
            }
            p = fitted.predict_p_left(make_stim_data(fitted.model, rows), max_draws=args.predict_max_draws)
            preds += [(i, name, f"{v:.6f}") for i, v in enumerate(p)]
            n_par.append([name, *n_params(fitted.model, n_participants)])
            if pool is not None:
                p = _pool_prediction(fitted, pool, model_name=name, participant_ids=pids)
                pool_preds += [(r["sequence_a"], r["sequence_b"], name, f"{v:.6f}") for r, v in zip(pool, p)]
            del fitted
            clear_fit_cache()

        # The comparison compare_table makes (az.compare's elpd_diff and dse,
        # the stimulus-clustered dse), over fits that may differ in seed.
        cmp = az.compare(loo, ic="loo")
        best = cmp.index[0]
        clusters = stimulus_clusters(rows)
        table = {
            name: {
                "rank": rank,
                "elpd_loo": float(row["elpd_loo"]),
                "elpd_diff": float(row["elpd_diff"]),
                "dse": float(row["dse"]),
                "dse_clustered": 0.0
                if name == best
                else cluster_dse(np.asarray(loo[best].loo_i), np.asarray(loo[name].loo_i), clusters),
                **info[name],
            }
            for rank, (name, row) in enumerate(cmp.iterrows())
        }
        (out / "compare.json").write_text(json.dumps(table, indent=2, default=float), encoding="utf-8")
        with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "model", "p_left"])
            w.writerows(preds)
        with (out / "n_params.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "n_free_rvs", "n_population_params", "n_person_level_vectors"])
            w.writerows(n_par)
        if pool is not None:
            with (out / "pool_predictions.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sequence_a", "sequence_b", "model", "p_left"])
                w.writerows(pool_preds)
        print(f"[fit] experiment {experiment}: wrote {out}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
