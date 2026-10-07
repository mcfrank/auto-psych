"""Did a recovery run find its held-out ground truth?

    uv run python -m src.rsa.recovery --loop-dir <run> --gt <gt.py> --test <sim_test.csv> --out-dir <dir>

A loop run on data simulated from a ground truth (src.rsa.simulate) recovers
it when the loop's final best model predicts like the ground truth. Two
measures, both with every model fitted on the run's training trials:

* prediction distance: RMSE between the model's and the ground truth's
  posterior-mean choice-class probabilities over the novelty pool (the
  novelty gate's measure; at the gate's 0.002 the two are the same
  hypothesis for the loop's purposes);
* held-out fit: the model's held-out lpd minus the ground truth's, with its
  clustered SE (src.rsa.evaluate_heldout).

Verdict: recovered when the best model is within ``rmse_threshold`` of the
ground truth, or its held-out lpd is within 2 SE of the ground truth's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tyro

from src.rsa.evaluate_heldout import Args as EvalArgs, main as evaluate
from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import fit_cached
from src.rsa.loop.novelty import closest, novelty_pool, posterior_mean_class_probs
from src.rsa.model_file import RSAModel
from src.runtime.config import PROJECT_ASSETS_DIR

DEFAULT_RMSE_THRESHOLD = 0.01


@dataclass
class Args:
    loop_dir: Path
    gt: Path
    test: Path
    out_dir: Path
    seed_models: Path = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
    rmse_threshold: float = DEFAULT_RMSE_THRESHOLD
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4


def main(args: Args) -> dict:
    loop = Path(args.loop_dir)
    export = json.loads((loop / "export.json").read_text())
    best = export["best_model"]
    table = evaluate(EvalArgs(loop_dir=loop, test=args.test, out_dir=args.out_dir, seed_models=args.seed_models,
                              num_warmup=args.num_warmup, num_samples=args.num_samples,
                              num_chains=args.num_chains, extra_models=[Path(args.gt)]))
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains)
    pool = novelty_pool()
    gt_model = RSAModel(args.gt, name=Path(args.gt).stem)
    gt_fit = fit_cached(args.gt, gt_model.name, loop / "responses.csv", settings, loop / ".fit_cache")
    gt_preds = posterior_mean_class_probs(gt_model, gt_fit, pool)
    distances = {}
    for name in export["live"]:
        path = loop / "models" / f"{name}.py"
        model = RSAModel(path, name=name)
        fitted = fit_cached(path, name, loop / "responses.csv", settings, loop / ".fit_cache")
        distances[name] = closest(posterior_mean_class_probs(model, fitted, pool), {"gt": gt_preds})[1]
    gt_label = f"extra:{Path(args.gt).stem}"
    lpd = dict(zip(table.model, table.lpd))
    pointwise = pd.read_csv(Path(args.out_dir) / "heldout_pointwise.csv")
    from src.models.clustered_se import cluster_dse
    from src.rsa.dataset import load_forced_choice
    from src.rsa.split import unit_keys

    units = pd.factorize(unit_keys(load_forced_choice(args.test).frame))[0]
    gap = lpd[best] - lpd[gt_label]
    se = cluster_dse(pointwise[gt_label].to_numpy(), pointwise[best].to_numpy(), units)
    verdict = dict(
        best_model=best, ground_truth=str(args.gt),
        best_pool_rmse_to_gt=distances[best], live_pool_rmse_to_gt=distances,
        heldout_lpd_best_minus_gt=gap, heldout_se=se,
        recovered=bool(distances[best] <= args.rmse_threshold or gap >= -2 * se),
        rmse_threshold=args.rmse_threshold,
    )
    (Path(args.out_dir) / "recovery.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    return verdict


if __name__ == "__main__":
    main(tyro.cli(Args))
