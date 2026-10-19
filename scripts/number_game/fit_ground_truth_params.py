"""Fit the number-game seeds to the whole dataset: the recovery runs' parameters.

    OMP_NUM_THREADS=1 uv run python -m scripts.number_game.fit_ground_truth_params

Each seed is fitted (warm-start NUTS) on every response of the Bigelow &
Piantadosi dataset, untrimmed, and its free parameters' posterior means are
printed as the holdout config's ``gt_models`` block: a ground truth simulates
at them, and the no-learning baseline scores every other seed at its own.
"""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import tyro
import yaml

from src.models.pymc_inference import convergence_problems_of, fit_model
from src.number_game import data as ng_data
from src.runtime.config import REPO_ROOT

DATA_CSV = ng_data.DATA_CSV
SEEDS_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "number_game" / "seed_models"


@dataclass
class Args:
    seeds: List[str] = field(
        default_factory=lambda: ["tenenbaum_rules_similarity", "bigelow_piantadosi_independent", "bigelow_piantadosi_interval"]
    )
    tune: int = 200
    draws: int = 500
    chains: int = 4


def main(args: Args) -> None:
    responses = ng_data.load_responses(DATA_CSV)
    params = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "responses.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["number_set", "target", "trial_index", "rating"])
            for i, (s, t, r) in enumerate(zip(responses["set"], responses["target"], responses["rating"])):
                writer.writerow([",".join(map(str, s)), t, i, r])
        for name in args.seeds:
            fitted = fit_model(
                name, SEEDS_DIR, path, method="warm_start_nuts",
                tune=args.tune, draws=args.draws, chains=args.chains, cores=args.chains,
            )
            problems = convergence_problems_of(fitted)
            if problems:
                raise RuntimeError(f"{name} did not converge on the whole dataset: {problems}")
            means = fitted.idata.posterior.mean(("chain", "draw"))
            params[name] = {
                rv.name: means[rv.name].values.round(6).tolist() for rv in fitted.model.free_RVs
            }
            print(f"[fit] {name}: converged, mean log-likelihood {fitted.mean_log_likelihood():.1f}", flush=True)
    print(yaml.safe_dump({"gt_models": params}, sort_keys=False, default_flow_style=None, width=100))


if __name__ == "__main__":
    main(tyro.cli(Args))
