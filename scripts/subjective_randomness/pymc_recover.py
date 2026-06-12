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
from src.subjective_randomness.tidy import (  # noqa: E402
    parameter_recovery_tidy_rows,
    write_tidy_csv,
)


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
    draws: Optional[int] = None
    """Posterior draws per chain (defaults to the config's `mcmc` block)."""
    tune: Optional[int] = None
    """Tuning (warmup) steps per chain (defaults to the config's `mcmc` block)."""
    chains: Optional[int] = None
    """Number of MCMC chains (defaults to the config's `mcmc` block)."""
    cores: Optional[int] = None
    """Number of cores for sampling (defaults to the config's `mcmc` block)."""
    tidy_csv: Optional[Path] = None
    """Optional long-format CSV (one row per parameter x repeat) for plotting."""


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
    if args.tidy_csv is not None:
        tidy_path = resolve_path(args.tidy_csv)
        write_tidy_csv(parameter_recovery_tidy_rows(result), tidy_path)
        print(f"Wrote tidy recovery CSV to {tidy_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
