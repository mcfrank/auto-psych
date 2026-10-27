"""Posterior summaries of the parameters of the paper's winning and seed models.

Reads the fits the figure-data scripts made (``sr_figure_data.py``,
``number_game_figure_data.py``) from their work directories on $SCRATCH and
samples nothing: each fit is found with ``load_cached_fit``, the lookup the
loop itself uses (same model files, data and sampler settings; a near miss's
refit when the fitting script refitted it). A missing fit raises.

Fit sets:

- ``sr_pooled``: every run's winner and the four seeds fitted on the three
  runs' final data pooled (Figure 3's fits).
- ``sr_run1`` .. ``sr_run3``: the run's winner and the seeds fitted on that
  run's final data at the live run's settings (Figure 2's per-run fits).
- ``number_game``: the run's winner and the three seeds fitted on all of its
  final data (not the CV folds).

Two tables are written to ``--out-dir``:

- ``population_parameters.csv``: one row per parameter (one per element of a
  vector parameter such as ``rule_weights[3]``) with its posterior mean, sd,
  94% HDI, bulk ESS and R-hat.
- ``person_level_parameters.csv``: one row per person-level parameter (a
  vector indexed by ``participant_id``), summarising the participants'
  posterior means: their mean, sd and quantiles, over the participants in the
  data only (a model has 400 slots; the rest are draws from the population
  distribution that no data reached), with the worst R-hat and bulk ESS
  among those slots.

Per-trial deterministics (``p_left``, ``p_yes``) are not parameters and are
left out.

Usage (loading the pooled fits takes ~10 GB of memory: run it as a job,
``sbatch scripts/paper/parameter_summaries.sbatch``):
    uv run python -m scripts.paper.parameter_summaries \\
        --sr-work-dir $SCRATCH/auto-psych/paper/sr_figure_data \\
        --number-game-work-dir $SCRATCH/auto-psych/paper/number_game_figure_data
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

SR_FIGURE_DATA = here() / "data" / "paper" / "figure_data" / "subjective_randomness.json"
SR_RUNS_ROOT = here() / "data" / "results" / "human_experiment_sr-oct26" / "collected"
NUMBER_GAME_ROOT = here() / "data" / "results" / "number_game_dataset-oct26"
DEFAULT_OUT_DIR = here() / "data" / "paper" / "parameter_summaries"

PERSON_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


@dataclass
class Args:
    sr_work_dir: Path
    """``sr_figure_data.py``'s work directory (``pooled/``, ``run<i>/``)."""
    number_game_work_dir: Path
    """``number_game_figure_data.py``'s work directory."""
    out_dir: Path = DEFAULT_OUT_DIR
    sr_figure_data: Path = SR_FIGURE_DATA
    """The SR figure data: which models were fitted, and at what settings."""
    sr_runs_root: Path = SR_RUNS_ROOT
    number_game_run_dir: Path = NUMBER_GAME_ROOT / "collected" / "run1" / "number_game"
    number_game_holdout: Path = NUMBER_GAME_ROOT / "harness" / "run1" / "holdout.json"
    number_game_figure_data: Path = here() / "data" / "paper" / "figure_data" / "number_game.json"


@dataclass
class FitSet:
    label: str
    roles: Dict[str, str]
    """Model name -> ``winner`` or ``seed``."""
    models_dir: Path
    responses: Path
    cache_dir: Path
    fit_kwargs: Dict[str, Any]


# --- which fits --------------------------------------------------------------


def sr_fit_sets(args: Args) -> List[FitSet]:
    figure = json.loads(args.sr_figure_data.read_text())
    pooled = figure["pooled"]
    pooled_dir = args.sr_work_dir / "pooled"
    fit_sets = [
        FitSet(
            label="sr_pooled",
            roles={name: ("winner" if name in pooled["winner_runs"] else "seed")
                   for name in pooled["elpd"]},
            models_dir=pooled_dir / "models",
            responses=pooled_dir / "responses.csv",
            cache_dir=pooled_dir / "fit_cache",
            fit_kwargs=pooled["fit_settings"],
        )
    ]
    for run in figure["runs"]:
        run_dir = args.sr_work_dir / run["run"]
        loop = (args.sr_runs_root / run["run"] / "subjective_randomness"
                / run["final_experiment"] / "model_loop")
        fit_sets.append(FitSet(
            label=f"sr_{run['run']}",
            roles={name: ("winner" if name == run["best_model"] else "seed")
                   for name in run["elpd"]},
            models_dir=run_dir / "models",
            responses=loop / "responses.csv",
            cache_dir=run_dir / "fit_cache",
            fit_kwargs=figure["fit_settings"],
        ))
    return fit_sets


def number_game_fit_set(args: Args) -> FitSet:
    winner = json.loads(args.number_game_figure_data.read_text())["runs"][0]["best_model"]
    seeds = json.loads((args.number_game_run_dir / "starting_models.json").read_text())[
        "starting_models"]
    return FitSet(
        label="number_game",
        roles={winner: "winner", **{seed: "seed" for seed in seeds}},
        models_dir=args.number_game_work_dir / "models",
        responses=args.number_game_work_dir / "responses.csv",
        cache_dir=args.number_game_work_dir / "fit_cache",
        fit_kwargs=json.loads(args.number_game_holdout.read_text())["fit_kwargs"],
    )


# --- summarising one fit -----------------------------------------------------


def participant_slots(idata: Any) -> Optional[np.ndarray]:
    """The person-level slots the data reached, or None for a model with no
    ``participant_id`` data."""
    if "constant_data" not in idata.groups() or "participant_id" not in idata.constant_data:
        return None
    return np.unique(idata.constant_data["participant_id"].values)


def is_person_level(shape: Tuple[int, ...], slots: Optional[np.ndarray]) -> bool:
    """A vector with a slot for every participant id in the data."""
    return slots is not None and len(shape) == 1 and shape[0] > slots.max()


def summarize_fit(idata: Any, n_trials: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(population rows, person-level rows) of one fit's posterior."""
    import arviz as az

    posterior = idata.posterior
    slots = participant_slots(idata)
    population_vars, person_rows = [], []
    for var in posterior.data_vars:
        shape = posterior[var].shape[2:]  # after (chain, draw)
        if n_trials in shape:
            continue  # a per-trial prediction, not a parameter
        if not is_person_level(shape, slots):
            population_vars.append(var)
            continue
        observed = posterior[[var]].isel({posterior[var].dims[2]: slots})
        means = observed[var].mean(("chain", "draw")).values
        person_rows.append({
            "parameter": var,
            "n_participants": len(slots),
            "mean_of_means": means.mean(),
            "sd_of_means": means.std(ddof=1),
            **{f"q{round(q * 100):02d}": np.quantile(means, q) for q in PERSON_QUANTILES},
            "max_r_hat": float(az.rhat(observed)[var].max()),
            "min_ess_bulk": float(az.ess(observed, method="bulk")[var].min()),
        })

    population = az.summary(posterior, var_names=population_vars, kind="all")
    population = population[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]]
    population = population.rename_axis("parameter").reset_index()
    return population, pd.DataFrame(person_rows)


# --- main --------------------------------------------------------------------


def summarize_fit_set(fit_set: FitSet) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    from src.models.pymc_inference import clear_fit_cache, load_cached_fit

    n_trials = len(pd.read_csv(fit_set.responses))
    population_tables, person_tables = [], []
    for name, role in fit_set.roles.items():
        print(f"[{fit_set.label}] {name} ({role})", flush=True)
        fitted = load_cached_fit(name, fit_set.models_dir, fit_set.responses,
                                 cache_dir=fit_set.cache_dir, **fit_set.fit_kwargs)
        population, person = summarize_fit(fitted.idata, n_trials)
        labels = {"fit_set": fit_set.label, "model": name, "role": role,
                  "fit_fingerprint": fitted.fingerprint}
        population_tables.append(population.assign(**labels))
        person_tables.append(person.assign(**labels))
        # A loaded fit holds its pointwise log-likelihood (~1 GB pooled).
        clear_fit_cache()
    return population_tables, person_tables


def main(args: Args) -> None:
    population_tables, person_tables = [], []
    for fit_set in [*sr_fit_sets(args), number_game_fit_set(args)]:
        population, person = summarize_fit_set(fit_set)
        population_tables += population
        person_tables += person

    label_columns = ["fit_set", "model", "role", "parameter"]
    population = pd.concat(population_tables, ignore_index=True)
    person = pd.concat(person_tables, ignore_index=True)
    population = population[label_columns + [c for c in population if c not in label_columns]]
    person = person[label_columns + [c for c in person if c not in label_columns]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    population.to_csv(args.out_dir / "population_parameters.csv", index=False)
    person.to_csv(args.out_dir / "person_level_parameters.csv", index=False)
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
