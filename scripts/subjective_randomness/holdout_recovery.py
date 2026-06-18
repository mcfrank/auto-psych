"""CLI: ground-truth holdout recovery through the full agentic loop.

For each configured seed model, hold it out as the ground truth (its
fixed-param PyMC model generates every synthetic response), run the full
agentic outer+inner loop seeded with the remaining models (real theory, design,
and candidate-conjecturing agents), and track — at every inner-loop scoring
step — the Pearson correlation between the then-best model and the ground
truth's ``p_left`` on a large held-out stimulus set.

This spawns real coding agents and runs real MCMC: a full 3-model run takes
hours. Use --gt-model / --n-experiments / --inner-loop-iterations to scope a
cheap smoke run first.

Usage:
    uv run python scripts/subjective_randomness/holdout_recovery.py \\
        --config scripts/subjective_randomness/configs/holdout_recovery.yaml \\
        --out data/subjective_randomness/holdout_recovery/holdout.json \\
        --tidy-csv data/subjective_randomness/holdout_recovery/holdout.csv \\
        --figure data/subjective_randomness/holdout_recovery/holdout.png
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import load_config, resolve_path  # noqa: E402
from src.subjective_randomness.holdout_recovery import (  # noqa: E402
    TRAJECTORY_COLUMNS,
    run_holdout_recovery_from_config,
    trajectory_tidy_rows,
)
from src.subjective_randomness.reporting import plot_holdout_trajectories  # noqa: E402
from src.subjective_randomness.tidy import write_tidy_csv  # noqa: E402


@dataclass
class Args:
    """Run ground-truth holdout recovery through the full agentic loop."""

    config: Path
    """YAML config (gt models, experiments, inner loop, eval pool, fit settings)."""
    out: Path
    """Output JSON path (combined result across held-out models)."""
    tidy_csv: Optional[Path] = None
    """Optional long-format CSV (one row per held-out model x trajectory step)."""
    figure: Optional[Path] = None
    """Optional correlation-vs-step plot (one line per held-out model)."""
    results_root: Optional[Path] = None
    """Where the per-model experiment trees go (default: <out dir>/<out stem>_runs)."""
    cache_dir: Optional[Path] = None
    """PyMC .nc fit cache shared by the run and the trajectory evaluation
    (default: <out dir>/mcmc_cache — the cache is what makes the per-step
    evaluation refits free)."""
    gt_model: Optional[str] = None
    """Hold out only this model (must be among the config's gt_models)."""
    gt_models_dir: Optional[Path] = None
    """Read the ground-truth generator(s) from this directory instead of the
    config's seed_models_dir. Use to keep the held-out GT file off the coding
    agent's sandbox: point this at a pristine copy of the seed models while
    deleting the held-out model from the agent's working checkout. The GT is
    still excluded from the agent's seed set (exclusion keys off the seed
    manifest, not this path)."""
    n_experiments: Optional[int] = None
    """Override the config's number of outer-loop experiments per run."""
    n_participants: Optional[int] = None
    """Override the config's synthetic participants per experiment."""
    inner_loop_iterations: Optional[int] = None
    """Override the config's inner-loop candidate rounds (0 = seed set only)."""
    inner_loop_candidates: Optional[int] = None
    """Override the config's candidate models per inner-loop round."""
    draws: Optional[int] = None
    """Override MCMC posterior draws per chain."""
    tune: Optional[int] = None
    """Override MCMC tuning (warmup) steps per chain."""
    chains: Optional[int] = None
    """Override the number of MCMC chains."""
    seed: Optional[int] = None
    """Override the config's RNG seed for the synthetic choices."""
    agent_timeout_sec: Optional[int] = None
    """Override the per-agent timeout in seconds."""
    backend: Optional[Literal["claude", "opencode"]] = None
    """Override the coding-agent backend (default: config, then CODING_AGENT env)."""
    resume: bool = False
    """Continue a stopped run: skip ground truths with a trajectory.json and,
    within incomplete runs, skip stages whose output already validates."""


def main(args: Args) -> None:
    config_path = resolve_path(args.config)
    out_path = resolve_path(args.out)
    results_root = (
        resolve_path(args.results_root)
        if args.results_root is not None
        else out_path.parent / f"{out_path.stem}_runs"
    )
    cache_dir = (
        resolve_path(args.cache_dir)
        if args.cache_dir is not None
        else out_path.parent / "mcmc_cache"
    )
    gt_models_dir = (
        resolve_path(args.gt_models_dir) if args.gt_models_dir is not None else None
    )

    fit_overrides = {
        key: value
        for key, value in (
            ("draws", args.draws),
            ("tune", args.tune),
            ("chains", args.chains),
        )
        if value is not None
    }
    inner_loop_overrides = {
        key: value
        for key, value in (
            ("max_iterations", args.inner_loop_iterations),
            ("candidate_count", args.inner_loop_candidates),
        )
        if value is not None
    }

    result = run_holdout_recovery_from_config(
        load_config(config_path),
        config_path,
        results_root,
        gt_model_override=args.gt_model,
        gt_models_dir=gt_models_dir,
        n_experiments_override=args.n_experiments,
        n_participants_override=args.n_participants,
        inner_loop_overrides=inner_loop_overrides or None,
        fit_overrides=fit_overrides or None,
        seed_override=args.seed,
        cache_dir=cache_dir,
        backend_override=args.backend,
        agent_timeout_override=args.agent_timeout_sec,
        resume=args.resume,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote holdout-recovery result to {out_path}")
    for gt_run in result["gt_runs"]:
        final = gt_run["trajectory"][-1]
        leakage = gt_run["leakage"]
        flags = ", ".join(
            name
            for name, on in (
                ("identical-copy", leakage["any_identical"]),
                ("gt-param-mention", leakage["any_mention"]),
                ("gt-named-file", leakage["any_gt_named"]),
            )
            if on
        )
        final_r = final["pearson_r"]
        print(
            f"  {gt_run['gt_model']}: {len(gt_run['trajectory'])} steps, "
            f"final r={'undefined' if final_r is None else f'{final_r:.3f}'} "
            f"(best={final['best_model']}, {gt_run['n_eval_stimuli']} eval stimuli, "
            f"{gt_run['n_eval_dropped']} dropped) "
            f"leakage flags: {flags or 'none'}"
        )

    if args.tidy_csv is not None:
        tidy_path = resolve_path(args.tidy_csv)
        write_tidy_csv(
            trajectory_tidy_rows(result), tidy_path, columns=TRAJECTORY_COLUMNS
        )
        print(f"Wrote tidy trajectory CSV to {tidy_path}")

    if args.figure is not None:
        figure_path = resolve_path(args.figure)
        plot_holdout_trajectories(result, figure_path)
        print(f"Wrote trajectory figure to {figure_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
