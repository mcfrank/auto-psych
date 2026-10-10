"""Pick a simulated run's hidden ground truth by rule.

``farthest_starting`` (the first rehearsal, 2026-10-09): the starting model the
promoted seeds represent least. It picked the literal listener, which the human
data reject by 205 lpd: the existing (human) and live (simulated) data then
came from incompatible processes, and the rehearsal could not inform the live
campaign (`docs/auto_rsa/REHEARSAL_REVIEW.md`).

``coherent`` (the second rehearsal, PI 2026-10-10): a model that fits the human
data about as well as the best ones, so existing and live data agree, but that
the run does not start with: among run 2's admitted models that were not
promoted, converged, within ``cv_window`` nats of the best on grouped CV over
all existing data (promotion.json), the one farthest (design-pool RMSE) from
its nearest seed of the chain the run starts from.

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
from typing import Literal, Optional

import numpy as np
import tyro

from src.models.model_manifest import read_manifest_names
from src.rsa.design.run import design_pool
from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC, loop_fit
from src.rsa.loop.novelty import posterior_mean_class_probs
from src.rsa.model_file import RSAModel
from src.rsa.outer.run import STARTING_MODELS


COHERENT_CV_WINDOW = 150.0  # nats behind the best run-2 model on grouped CV (fixed 2026-10-10)


@dataclass
class Args:
    promoted: Path
    data: Path
    cache: Path
    out: Path
    starting_models: Path = STARTING_MODELS
    rule: Literal["farthest_starting", "coherent"] = "farthest_starting"
    promotion: Optional[Path] = None
    """coherent: promote's record (data/rsa/live_seeds/promotion.json)."""
    sweep: Optional[Path] = None
    """coherent: run 2's brought-back sweep (its cells' model files)."""
    chain: Optional[Path] = None
    """coherent: the seeds the run starts from (data/rsa/live_seeds/chains/chain_<k>)."""
    cv_window: float = COHERENT_CV_WINDOW
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


def coherent(args: Args, settings: FitSettings) -> dict:
    from src.rsa.promote import candidates

    if args.promotion is None or args.sweep is None or args.chain is None:
        raise ValueError("the coherent rule needs --promotion, --sweep and --chain")
    record = json.loads(Path(args.promotion).read_text())
    eligible = [c for c in record["candidates"] if not c["chosen"] and c["converged"] and c["cv_converged"]
                and c["cv_behind_best"] is not None and c["cv_behind_best"] <= args.cv_window]
    if not eligible:
        raise ValueError(f"no unpromoted run-2 model within {args.cv_window} nats of the best on grouped CV")
    files = {c.key: c for c in candidates(Path(args.sweep), record["cells"])}
    pool = design_pool()
    seeds = pool_predictions(args.chain, args.data, args.cache, settings)
    rows = {}
    for c in eligible:
        cand = files[c["key"]]
        if cand.sha != c["sha256"]:
            raise ValueError(f"{c['key']}: its file is not the one promote scored")
        fitted = loop_fit(cand.path, cand.name, args.data, settings, args.cache, time_limit_sec=FIT_TIME_LIMIT_SEC)
        p = posterior_mean_class_probs(RSAModel(cand.path, name=cand.name), fitted, pool)
        d = {m: float(np.sqrt(np.mean((p - q) ** 2))) for m, q in seeds.items()}
        nearest = min(d, key=d.get)
        rows[c["key"]] = dict(file=str(cand.path), cv_behind_best=c["cv_behind_best"], nearest_seed=nearest,
                              rmse=d[nearest])
    key = max(rows, key=lambda k: (rows[k]["rmse"], k))
    return dict(ground_truth=Path(rows[key]["file"]).stem, ground_truth_key=key, ground_truth_file=rows[key]["file"],
                rule=f"coherent: among run 2's unpromoted, converged models within {args.cv_window} nats of the best "
                     "on grouped CV, the one farthest (design-pool RMSE) from its nearest seed of the chain",
                chain=str(args.chain), candidates=rows)


def main(args: Args) -> dict:
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.fit_seed)
    if args.rule == "coherent":
        record = coherent(args, settings)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=1))
        print(json.dumps({k: v for k, v in record.items() if k != "candidates"}, indent=1))
        return record
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
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()  # one core per process (src.rsa.cpus)
    main(tyro.cli(Args))
