"""CLI: run PyMC parameter recovery for a subjective-randomness model family.

Usage:
    uv run python scripts/subjective_randomness/pymc_recover.py \\
        --config CONFIG.yaml --out REPORT.json [--n-repeats 20]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import load_config, resolve_path  # noqa: E402
from src.subjective_randomness.pymc_recover import PYMC_MODELS_DIR, run_pymc_recovery  # noqa: E402


@dataclass
class Args:
    """Run PyMC parameter recovery for a subjective-randomness model family."""

    config: Path
    """YAML config."""
    out: Path
    """Output JSON report path."""
    n_repeats: Optional[int] = None
    """Number of simulate-and-fit repeats (defaults to the config value)."""
    pymc_models_dir: Path = PYMC_MODELS_DIR
    """Directory holding the PyMC adapter modules."""
    work_dir: Optional[Path] = None
    """Optional directory to keep simulated featurized CSVs."""
    cache_dir: Optional[Path] = None
    """Optional directory for the PyMC .nc fit cache."""
    draws: int = 500
    """Posterior draws per chain."""
    tune: int = 500
    """Tuning (warmup) steps per chain."""
    chains: int = 2
    """Number of MCMC chains."""
    cores: int = 1
    """Number of cores for sampling."""


def main(args: Args) -> None:
    config_path = resolve_path(args.config)
    result = run_pymc_recovery(
        load_config(config_path),
        config_path,
        repeats_override=args.n_repeats,
        pymc_models_dir=resolve_path(args.pymc_models_dir),
        work_dir=resolve_path(args.work_dir) if args.work_dir else None,
        cache_dir=resolve_path(args.cache_dir) if args.cache_dir else None,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
    )
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Wrote PyMC recovery report for {result['model']} "
        f"({result['n_repeats']} repeats, {result['n_stimuli']} stimuli) to {out_path}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
