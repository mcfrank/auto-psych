"""Did a recovery run find its held-out ground truth?

    uv run python -m src.rsa.recovery --loop-dir <run> --gt <gt.py> --test <sim_test.csv> --out-dir <dir>

A loop run on data simulated from a ground truth (src.rsa.simulate) recovers
it when the loop's exported model predicts like the ground truth: the RMSE
between the two models' posterior-mean choice-class probabilities over the
novelty pool (the novelty gate's measure) is at most ``rmse_threshold``.
Every model is fitted on the run's training trials with the loop's fit rule
(its cache is reused).

Reported beside the verdict, not part of it:

* the closest live model's distance (in all four recovery cells of Sherlock
  run 1 a live model was closer than the exported one);
* the starting models' distances, the baseline: a verdict the seeds would
  pass says nothing;
* the held-out fit, the exported model's held-out lpd minus the ground
  truth's with its clustered SE (src.rsa.evaluate_heldout). It used to be an
  "or" clause of the verdict and was powerless: on the salience data rsa_l1
  and rsa_l2, 0.05 RMSE from the truth, were within 2 SE of it on held-out
  conditions (Sherlock run 1, PI decision 2026-10-08).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro

from src.models.model_manifest import read_manifest_names
from src.rsa.evaluate_heldout import Args as EvalArgs, main as evaluate, seed_models_dir
from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import loop_fit
from src.rsa.loop.novelty import closest, novelty_pool, posterior_mean_class_probs
from src.rsa.model_file import RSAModel

DEFAULT_RMSE_THRESHOLD = 0.01


@dataclass
class Args:
    loop_dir: Path
    gt: Path
    test: Path
    out_dir: Path
    seed_models: Optional[Path] = None
    """Baseline seeds (default: the run's seed_pool/, see src.rsa.evaluate_heldout)."""
    rmse_threshold: float = DEFAULT_RMSE_THRESHOLD
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0
    """The loop run's --seed: with the same NUTS settings, every fit is the
    loop's own (read from its cache, refit included; src.rsa.loop.fitting.loop_fit)."""


def main(args: Args) -> dict:
    loop = Path(args.loop_dir)
    export = json.loads((loop / "export.json").read_text())
    best = export["best_model"]
    table = evaluate(EvalArgs(loop_dir=loop, test=args.test, out_dir=args.out_dir, seed_models=args.seed_models,
                              num_warmup=args.num_warmup, num_samples=args.num_samples,
                              num_chains=args.num_chains, seed=args.seed, extra_models=[Path(args.gt)]))
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.seed)
    pool = novelty_pool()
    gt_model = RSAModel(args.gt, name=Path(args.gt).stem)
    gt_fit = loop_fit(args.gt, gt_model.name, loop / "responses.csv", settings, loop / ".fit_cache")
    gt_preds = posterior_mean_class_probs(gt_model, gt_fit, pool)
    distances = {}
    for name in export["live"]:
        path = loop / "models" / f"{name}.py"
        model = RSAModel(path, name=name)
        fitted = loop_fit(path, name, loop / "responses.csv", settings, loop / ".fit_cache")
        distances[name] = closest(posterior_mean_class_probs(model, fitted, pool), {"gt": gt_preds})[1]
    seeds = seed_models_dir(EvalArgs(loop_dir=loop, test=args.test, out_dir=args.out_dir, seed_models=args.seed_models))
    seed_distances = {}
    for name in read_manifest_names(seeds):
        path = seeds / f"{name}.py"
        model = RSAModel(path, name=name)
        fitted = loop_fit(path, name, loop / "responses.csv", settings, loop / ".fit_cache")
        seed_distances[name] = closest(posterior_mean_class_probs(model, fitted, pool), {"gt": gt_preds})[1]
    gt_label = f"extra:{Path(args.gt).stem}"
    lpd = dict(zip(table.model, table.lpd))
    pointwise = pd.read_csv(Path(args.out_dir) / "heldout_pointwise.csv")
    from src.models.clustered_se import cluster_dse
    from src.rsa.dataset import load_forced_choice
    from src.rsa.split import unit_keys

    units = pd.factorize(unit_keys(load_forced_choice(args.test).frame))[0]
    gap = lpd[best] - lpd[gt_label]
    se = cluster_dse(pointwise[gt_label].to_numpy(), pointwise[best].to_numpy(), units)
    closest_live = min(distances, key=distances.get)
    best_seed = min(seed_distances, key=seed_distances.get)
    verdict = dict(
        best_model=best, ground_truth=str(args.gt),
        best_pool_rmse_to_gt=distances[best], live_pool_rmse_to_gt=distances,
        closest_live_model=closest_live, closest_live_rmse_to_gt=distances[closest_live],
        seed_pool_rmse_to_gt=seed_distances, best_seed=best_seed, best_seed_rmse_to_gt=seed_distances[best_seed],
        seeds_would_pass=bool(seed_distances[best_seed] <= args.rmse_threshold),
        heldout_lpd_best_minus_gt=gap, heldout_se=se,
        recovered=bool(distances[best] <= args.rmse_threshold),
        rmse_threshold=args.rmse_threshold,
        verdict_rule="exported model's pool RMSE to the ground truth <= rmse_threshold",
    )
    (Path(args.out_dir) / "recovery.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    return verdict


if __name__ == "__main__":
    main(tyro.cli(Args))
