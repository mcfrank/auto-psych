"""CLI: re-score one finished holdout-recovery run on the EXHAUSTIVE eval pool.

Some holdout-recovery studies measured recovery on a *sampled* held-out pool
(e.g. 500 pairs at lengths 6 & 8) while others used the *exhaustive* pool (every
distinct unordered pair over all sequences up to length 8). That makes their
RMSE / Pearson-r numbers incomparable. This CLI re-analyzes a single finished
run's ``holdout.json`` so it is scored on the common exhaustive pool instead,
without re-running any agents or resampling MCMC:

* it rebuilds the held-out stimulus set as the exhaustive pair space over the
  requested ``--lengths`` (default 1..8), still excluding the pairs that run
  actually trained on, then
* re-correlates every inner-loop trajectory step and both seed baselines against
  the ground truth on that pool, reusing the run's MCMC fit cache (the per-step
  refits are cache hits, so the only real cost is predicting over the ~130k-pair
  pool — ``--predict-max-draws`` thins the posterior to keep that bounded).

The ground-truth generator is located automatically: a normal seed model is read
from ``--seed-models-dir`` (the canonical project seed dir, which still holds the
held-out model — the agent's sandbox copy had it deleted, this one does not); an
impossible ground truth is read from ``src/subjective_randomness/impossible_models``.
Override with ``--gt-models-dir`` if needed.

By default the enriched JSON / CSV / figures overwrite the run's ``holdout.json``
in place, so the existing test-retest and combined-plot collectors pick the
exhaustive-pool numbers up unchanged.

Usage:
    uv run python scripts/subjective_randomness/reanalyze_holdout_exhaustive.py \\
        --result $SCRATCH/auto-psych/holdout_test_retest/run1/window_typicality/holdout.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.pipelines.outer_loop.orchestrator import (  # noqa: E402
    project_seed_models_dir,
)
from src.subjective_randomness.config import resolve_path  # noqa: E402
from src.subjective_randomness.holdout_recovery import (  # noqa: E402
    PROJECT_ID,
    TRAJECTORY_COLUMNS,
    reevaluate_trajectories,
    trajectory_tidy_rows,
)
from src.subjective_randomness.reporting import plot_holdout_trajectories  # noqa: E402
from src.subjective_randomness.tidy import write_tidy_csv  # noqa: E402

IMPOSSIBLE_MODELS_DIR = here() / "src/subjective_randomness/impossible_models"


@dataclass
class Args:
    """Re-score a finished holdout-recovery run on the exhaustive eval pool."""

    result: Path
    """The run's holdout-recovery result JSON to re-analyze."""
    lengths: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    """Sequence lengths spanning the exhaustive held-out pool (every distinct
    unordered pair over all sequences of these lengths, cross-length included)."""
    predict_max_draws: int = 500
    """Thin the posterior to this many draws when predicting held-out p_left
    (keeps the draws x n_stim array bounded over the ~130k-pair pool)."""
    min_remaining: int = 100
    """Fail loudly if fewer than this many eval pairs survive trained-pair
    exclusion."""
    out: Optional[Path] = None
    """Where to write the enriched result JSON (default: overwrite --result)."""
    figure: Optional[Path] = None
    """Output figure path (default: <result dir>/<result stem>.png)."""
    tidy_csv: Optional[Path] = None
    """Output tidy CSV path (default: <result dir>/<result stem>.csv)."""
    cache_dir: Optional[Path] = None
    """Shared PyMC .nc fit cache (default: <result dir>/mcmc_cache). The cache
    is what makes the per-step refits free."""
    seed_models_dir: Optional[Path] = None
    """Seed-model directory holding the seed generators AND a normal ground
    truth (default: the canonical project seed dir, which — unlike the agent's
    sandbox copy — still contains every held-out model)."""
    gt_models_dir: Optional[Path] = None
    """Directory holding the ground-truth generator. Default: auto — the seed
    dir for a normal GT, the impossible-models dir for an impossible GT."""


def _resolve_gt_models_dir(
    result: dict, seed_models_dir: Path, override: Optional[Path]
) -> Path:
    """Where the ground-truth generator lives for this run.

    Honors an explicit override, else picks the seed dir for a normal ground
    truth and the impossible-models dir for an impossible one, failing loudly if
    the generator source is in neither.
    """
    if override is not None:
        return resolve_path(override)
    gt_model = result["gt_runs"][0]["gt_model"]
    if (seed_models_dir / f"{gt_model}.py").exists():
        return seed_models_dir
    if (IMPOSSIBLE_MODELS_DIR / f"{gt_model}.py").exists():
        return IMPOSSIBLE_MODELS_DIR
    raise FileNotFoundError(
        f"Ground-truth generator {gt_model!r} not found in the seed dir "
        f"({seed_models_dir}) or the impossible-models dir "
        f"({IMPOSSIBLE_MODELS_DIR}); pass --gt-models-dir explicitly."
    )


def main(args: Args) -> None:
    result_path = resolve_path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    cache_dir = (
        resolve_path(args.cache_dir)
        if args.cache_dir is not None
        else result_path.parent / "mcmc_cache"
    )
    seed_models_dir = (
        resolve_path(args.seed_models_dir)
        if args.seed_models_dir is not None
        else project_seed_models_dir(result.get("project_id", PROJECT_ID))
    )
    gt_models_dir = _resolve_gt_models_dir(result, seed_models_dir, args.gt_models_dir)

    eval_pool_override = {
        "exhaustive": True,
        "lengths": list(args.lengths),
        "predict_max_draws": args.predict_max_draws,
        "min_remaining": args.min_remaining,
    }
    print(
        f"Re-scoring {result_path} on the exhaustive pool over lengths "
        f"{eval_pool_override['lengths']} (gt_models_dir={gt_models_dir})"
    )

    enriched = reevaluate_trajectories(
        result,
        seed_models_dir=seed_models_dir,
        cache_dir=cache_dir,
        gt_models_dir=gt_models_dir,
        eval_pool_override=eval_pool_override,
    )

    out_path = resolve_path(args.out) if args.out is not None else result_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    print(f"Wrote enriched holdout-recovery result to {out_path}")

    def _fmt(value: object) -> str:
        return "undefined" if value is None else f"{value:.3f}"

    for gt_run in enriched["gt_runs"]:
        final = gt_run["trajectory"][-1]
        print(
            f"  {gt_run['gt_model']}: n_eval={gt_run['n_eval_stimuli']} "
            f"(dropped {gt_run['n_eval_dropped']}) | final best r={_fmt(final['pearson_r'])} "
            f"rmse={_fmt(final['rmse'])} (best={final['best_model']}), "
            f"BMA r={_fmt(final['pearson_r_bma'])} rmse={_fmt(final['rmse_bma'])}, "
            f"fitted-seed r={_fmt(gt_run['fitted_baseline']['mean_r'])}, "
            f"default-seed r={_fmt(gt_run['baseline']['mean_r'])}"
        )

    tidy_path = (
        resolve_path(args.tidy_csv)
        if args.tidy_csv is not None
        else result_path.with_suffix(".csv")
    )
    write_tidy_csv(trajectory_tidy_rows(enriched), tidy_path, columns=TRAJECTORY_COLUMNS)
    print(f"Wrote tidy trajectory CSV to {tidy_path}")

    figure_path = (
        resolve_path(args.figure)
        if args.figure is not None
        else result_path.with_suffix(".png")
    )
    plot_holdout_trajectories(enriched, figure_path, metric="pearson_r")
    print(f"Wrote trajectory figure (Pearson r) to {figure_path}")

    rmse_figure_path = figure_path.with_name(
        f"{figure_path.stem}_rmse{figure_path.suffix}"
    )
    plot_holdout_trajectories(enriched, rmse_figure_path, metric="rmse")
    print(f"Wrote trajectory figure (RMSE) to {rmse_figure_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
