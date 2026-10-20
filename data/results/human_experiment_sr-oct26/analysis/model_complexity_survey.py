"""Every model each experiment's loop scored: its ELPD margin over the best seed and its size.

Are there models, earlier in a run, that beat the seeds decisively without
the winners' accreted cue bundles? For each (run, experiment) this takes the
models in the last step of ``model_loop/history.json`` (every model the loop
had at the end, before pruning) with their ELPD-LOO, and counts each model's
size from its PyMC graph: stimulus inputs (``pm.Data`` other than
``participant_id`` / ``chose_left``), population-level scalars, and
person-level vectors (a free variable with one entry per participant slot).

The best seed's ELPD on the same data: experiment 1 from the loop's own step 0
(the seeds were scored there); experiments 2-3 from this analysis's refits
(``BEST_SEED``; the loop pruned every seed in experiment 1).

Writes ``results/model_complexity_survey.csv``.

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/model_complexity_survey.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

COLLECTED = here() / "data/results/human_experiment_sr-oct26/collected"
ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"
SEEDS = ("falk_konold_dp", "finite_experience_occurrence", "local_representativeness", "motif_stack")
# Best seed ELPD-LOO on experiment N's cumulative data, from this analysis's
# refits: run 1 fits/experiment{2,3}/compare.json; run 2 the person-level
# heldout (experiment 2) and all (experiment 3) fits. local_representativeness
# in every case.
BEST_SEED = {(1, 2): -3002.0, (1, 3): -4538.7, (2, 2): -2891.4, (2, 3): -4385.4}
BOOKKEEPING = {"participant_id", "chose_left", "trial_index"}


@dataclass
class Args:
    """Survey the ELPD and size of every model the loop scored."""

    runs: tuple[int, ...] = (1, 2)
    out: Path = ANALYSIS_ROOT / "results/model_complexity_survey.csv"


def find_source(root: Path, experiment: int, name: str) -> Path:
    for e in range(experiment, 0, -1):
        for d in ("model_loop/models", "model_loop/models/pruned", "cognitive_models_input"):
            path = root / f"experiment{e}" / d / f"{name}.py"
            if path.exists():
                return path
    raise FileNotFoundError(f"{name} (experiment {experiment}) not found under {root}")


def size_of(path: Path) -> dict:
    from src.models.model_loading import load_pymc_model, pm_data_inputs

    model = load_pymc_model(path.stem, path.parent)
    point = model.initial_point()
    sizes = [int(np.prod(point[model.rvs_to_values[rv].name].shape)) for rv in model.free_RVs]
    return {
        "n_inputs": len([d for d in pm_data_inputs(model) if d not in BOOKKEEPING]),
        "n_population_params": sum(1 for n in sizes if n < 100),
        "n_person_vectors": sum(1 for n in sizes if n >= 100),
    }


def first_step(history: list[dict], name: str) -> int:
    return min(s["step"] for s in history if name in s["elpd_loo"])


def main(args: Args) -> None:
    rows = []
    for run in args.runs:
        root = COLLECTED / f"run{run}/subjective_randomness"
        for experiment in (1, 2, 3):
            history = json.loads((root / f"experiment{experiment}/model_loop/history.json").read_text())
            elpd = history[-1]["elpd_loo"]
            best_seed = (
                max(history[0]["elpd_loo"][s] for s in SEEDS) if experiment == 1 else BEST_SEED[(run, experiment)]
            )
            top = max(elpd.values())
            for name, value in elpd.items():
                rows.append(
                    {
                        "run": run,
                        "experiment": experiment,
                        "model": name,
                        "seed": name in SEEDS,
                        "first_step": first_step(history, name),
                        "elpd_loo": value,
                        "over_best_seed": value - best_seed,
                        "behind_best": top - value,
                        **size_of(find_source(root, experiment, name)),
                    }
                )
    table = pd.DataFrame(rows).sort_values(["run", "experiment", "elpd_loo"], ascending=[True, True, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.to_string(index=False, float_format="%.1f"))


if __name__ == "__main__":
    main(tyro.cli(Args))
