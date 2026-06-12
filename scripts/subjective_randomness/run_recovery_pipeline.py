"""CLI: run the whole subjective-randomness recovery pipeline.

Runs parameter recovery for every model-family config (ground truths sampled
across the parameter space by default), the stimulus-selection comparison
(the same grid-posterior recovery on an EIG-optimized vs. a random stimulus
set, paired on the same truths), and closed-ended model recovery (each stage
skippable), and writes all reports, summary CSVs, and figures plus an
aggregated `key_results.txt` into one output directory.

Usage:
    uv run python scripts/subjective_randomness/run_recovery_pipeline.py
    # quick parameter-recovery-only check:
    uv run python scripts/subjective_randomness/run_recovery_pipeline.py \\
        --out-dir /tmp/recovery_smoke --n-repeats 5 --skip-model-recovery
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import resolve_path  # noqa: E402
from src.subjective_randomness.pipeline import run_pipeline  # noqa: E402

CONFIG_DIR = Path("scripts/subjective_randomness/configs")
DEFAULT_PARAM_CONFIGS = (
    CONFIG_DIR / "prototype_similarity.yaml",
    CONFIG_DIR / "encoding_compressibility.yaml",
    CONFIG_DIR / "bayesian_diagnosticity.yaml",
)


@dataclass
class Args:
    """Run the full recovery pipeline (parameter recovery + model recovery)."""

    out_dir: Path = Path("data/subjective_randomness/recovery_pipeline")
    """Directory for all pipeline artifacts (reports, figures, key_results.txt)."""
    param_configs: Tuple[Path, ...] = DEFAULT_PARAM_CONFIGS
    """Parameter-recovery YAML configs, one per model family."""
    model_recovery_config: Path = CONFIG_DIR / "model_recovery.yaml"
    """Closed-ended model-recovery YAML config."""
    skip_model_recovery: bool = False
    """Skip the model-recovery stage (it fits every seed model with MCMC)."""
    selection_comparison_config: Path = CONFIG_DIR / "selection_comparison.yaml"
    """EIG-optimized vs. random stimulus-set comparison YAML config."""
    skip_selection_comparison: bool = False
    """Skip the EIG-optimized vs. random stimulus-set comparison stage."""
    n_repeats: Optional[int] = None
    """Override each parameter-recovery config's repeat count."""
    n_participants: Optional[int] = None
    """Override the model-recovery config's synthetic participants."""
    draws: Optional[int] = None
    """Override MCMC posterior draws per chain (parameter and model recovery)."""
    tune: Optional[int] = None
    """Override MCMC tuning (warmup) steps per chain (parameter and model recovery)."""
    chains: Optional[int] = None
    """Override the number of MCMC chains (parameter and model recovery)."""


def main(args: Args) -> None:
    fit_overrides = {
        key: value
        for key, value in (
            ("draws", args.draws),
            ("tune", args.tune),
            ("chains", args.chains),
        )
        if value is not None
    }
    out_dir = resolve_path(args.out_dir)

    result = run_pipeline(
        [resolve_path(p) for p in args.param_configs],
        out_dir,
        None if args.skip_model_recovery else resolve_path(args.model_recovery_config),
        selection_comparison_config_path=(
            None
            if args.skip_selection_comparison
            else resolve_path(args.selection_comparison_config)
        ),
        n_repeats=args.n_repeats,
        n_participants=args.n_participants,
        fit_overrides=fit_overrides or None,
    )

    print("\n" + result["key_results_path"].read_text(encoding="utf-8"))
    print(f"Key results written to {result['key_results_path']}")
    print(f"All pipeline artifacts in {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
