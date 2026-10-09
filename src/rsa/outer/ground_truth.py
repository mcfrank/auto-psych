"""Pick a simulated run's hidden ground truth by rule: the starting model the
promoted seeds represent least.

    uv run python -m src.rsa.outer.ground_truth --promoted data/rsa/live_seeds/models \
        --data <all existing trials> --cache <fit cache> --out <private dir>/ground_truth.json

Each starting model (`src.rsa.outer.run.STARTING_MODELS`) and each promoted
seed is fitted to the existing data (promote's fits are reused from its cache);
the ground truth is the starting model whose posterior-mean choice-class
probabilities on the design pool are farthest (RMSE) from its nearest promoted
seed. A truth the live phase does not start with is one it has to find, and the
existing data (real, not simulated) pull toward the models that fit them: the
setting where cumulative and live-only selection can differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from src.models.model_manifest import read_manifest_names
from src.rsa.design.run import design_pool
from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC, loop_fit
from src.rsa.loop.novelty import posterior_mean_class_probs
from src.rsa.model_file import RSAModel
from src.rsa.outer.run import STARTING_MODELS


@dataclass
class Args:
    promoted: Path
    data: Path
    cache: Path
    out: Path
    starting_models: Path = STARTING_MODELS
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    fit_seed: int = 0


def pool_predictions(folder: Path, data: Path, cache: Path, settings: FitSettings) -> dict:
    pool = design_pool()
    out = {}
    for name in read_manifest_names(folder):
        path = Path(folder) / f"{name}.py"
        fitted = loop_fit(path, name, data, settings, cache, time_limit_sec=FIT_TIME_LIMIT_SEC)
        out[name] = posterior_mean_class_probs(RSAModel(path, name=name), fitted, pool)
    return out


def main(args: Args) -> dict:
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.fit_seed)
    promoted = pool_predictions(args.promoted, args.data, args.cache, settings)
    starting = pool_predictions(args.starting_models, args.data, args.cache, settings)
    nearest = {}
    for s, p in starting.items():
        d = {m: float(np.sqrt(np.mean((p - q) ** 2))) for m, q in promoted.items()}
        m = min(d, key=d.get)
        nearest[s] = dict(nearest_promoted=m, rmse=d[m])
    gt = max(nearest, key=lambda s: nearest[s]["rmse"])
    record = dict(ground_truth=gt, ground_truth_file=str(Path(args.starting_models) / f"{gt}.py"),
                  rule="the starting model farthest (design-pool RMSE) from its nearest promoted seed",
                  starting_models=nearest)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))
    return record


if __name__ == "__main__":
    main(tyro.cli(Args))
