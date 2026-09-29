"""CLI: score the current best model of an IN-PROGRESS holdout cell.

A holdout cell is only scored when it finishes. This scores a cell mid-run:
the incumbent of its latest experiment with an inner-loop step, and the
fitted-seed baseline (every seed model other than the ground truth, from the
files the cell was seeded with), all fit on
that experiment's inner-loop responses, which accumulate every experiment's
data so far. Held-out RMSE / Pearson r against the ground truth are computed on
the exhaustive pool minus the pairs trained on so far, as the harness does at
the end (the harness also excludes the pairs of experiments not yet run).

Nothing is written into the live run: the run's fit cache is reached through
links in a private cache directory, where any new fit lands.

Usage:
    python scripts/subjective_randomness/snapshot_recovery.py \\
        --run-dir $WORK_ROOT/run1/motif_stack \\
        --config scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml \\
        --gt-models-dir $WORK_ROOT/gt_models_src --gt-family-dir $WORK_ROOT/gt_family_src \\
        --out $SCRATCH/auto-psych/partial_recovery/sweep_run1_motif_stack.json
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tyro
import yaml
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.models.pymc_inference import fit_model  # noqa: E402
from src.subjective_randomness.holdout_data import (  # noqa: E402
    _raw_eval_rows,
    p_left_fixed_params,
    resolve_generating_params,
    seed_model_names,
)
from src.subjective_randomness.holdout_eval import (  # noqa: E402
    _eval_prediction,
    _participant_ids_in,
    _resolve_model_dir,
    build_eval_stimuli,
    seeded_models_dir,
)
from src.subjective_randomness.recover import pearson_r  # noqa: E402


@dataclass
class Args:
    run_dir: Path
    """The cell's run directory, $WORK_ROOT/run<r>/<gt> (holds the repo link and mcmc_cache)."""
    config: Path
    """The holdout config the cell was launched with."""
    gt_models_dir: Path
    """The pristine registry the setup job staged ($WORK_ROOT/gt_models_src)."""
    gt_family_dir: Path
    """The pristine model families ($WORK_ROOT/gt_family_src), for the GT's default params."""
    out: Path
    """Where to write the JSON result."""
    chains: int = 4
    """MCMC chains, as the array job passes them (part of the fit-cache key)."""


def latest_loop(tree: Path) -> Tuple[Path, int]:
    """The model_loop dir of the highest-numbered experiment with a recorded step."""
    loops = []
    for history in Path(tree).glob("experiment*/model_loop/history.json"):
        exp_num = int(re.fullmatch(r"experiment(\d+)", history.parts[-3]).group(1))
        if json.loads(history.read_text(encoding="utf-8")):
            loops.append((exp_num, history.parent))
    if not loops:
        raise FileNotFoundError(f"{tree} has no inner-loop step yet")
    exp_num, loop_dir = max(loops)
    return loop_dir, exp_num


def private_cache(run_cache: Path, cache_dir: Path) -> Path:
    """A cache dir holding links to every fit in ``run_cache``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fit in run_cache.glob("*.nc"):
        link = cache_dir / fit.name
        if not link.exists():
            link.symlink_to(fit.resolve())
    return cache_dir


def main(args: Args) -> None:
    gt_model = args.run_dir.name
    tree = (args.run_dir / "repo").resolve() / "_runs" / "cell_1"
    loop_dir, exp_num = latest_loop(tree)
    history = json.loads((loop_dir / "history.json").read_text(encoding="utf-8"))
    incumbent = history[-1]["best_model"]

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gt_params = resolve_generating_params(
        config.get("gt_models"), args.gt_models_dir, args.gt_family_dir
    )[gt_model]
    fit_kwargs = {**dict(config.get("fit", {})), "chains": args.chains}
    pool = dict(config.get("eval_pool", {}))
    eval_info = build_eval_stimuli(
        tree,
        n_experiments=exp_num,
        n_pairs=int(pool.get("n_pairs", 500)),
        lengths=[int(x) for x in pool.get("lengths", (6, 8))],
        seed=int(pool.get("seed", 11)),
        min_remaining=int(pool.get("min_remaining", 100)),
        exhaustive=bool(pool.get("exhaustive", True)),
    )
    stimuli = eval_info["stimuli"]
    gt_p = p_left_fixed_params(gt_model, args.gt_models_dir, stimuli, gt_params)
    eval_rows = _raw_eval_rows(stimuli)
    responses = loop_dir / "responses.csv"
    participant_ids = _participant_ids_in(responses)
    cache_dir = private_cache(
        args.run_dir / "mcmc_cache", args.out.parent / "cache" / args.out.stem
    )

    def score(name: str, models_dir: Path) -> Dict[str, Any]:
        fitted = fit_model(name, models_dir, responses, cache_dir=cache_dir, **fit_kwargs)
        pred = _eval_prediction(
            fitted, eval_rows, participant_ids=participant_ids,
            max_draws=pool.get("predict_max_draws"),
        )
        return {
            "rmse": float(np.sqrt(np.mean((gt_p - pred) ** 2))),
            "pearson_r": pearson_r(gt_p.tolist(), pred.tolist()),
        }

    result = {
        "gt_model": gt_model,
        "run_dir": str(args.run_dir),
        "experiment": exp_num,
        "step": len(history) - 1,
        "n_eval_stimuli": len(stimuli),
        "incumbent": {
            "name": incumbent,
            **score(incumbent, _resolve_model_dir(loop_dir / "models", incumbent)),
        },
        # The seed files the cell was seeded with, not the registry's, which
        # may have changed since.
        "seeds": {
            name: score(name, seeded_models_dir(tree))
            for name in seed_model_names(args.gt_models_dir)
            if name != gt_model
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
