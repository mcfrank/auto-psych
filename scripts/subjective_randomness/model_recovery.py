"""CLI: run closed-ended model recovery for the subjective-randomness seed set.

Generate synthetic data from each seed model (parameters fixed), run the inner
model loop on the *closed* seed set (no agent candidates), and write the
recovered generating-model x recovered-model confusion matrix as JSON plus a
tidy long-format CSV for plotting.

Usage:
    uv run python scripts/subjective_randomness/model_recovery.py \\
        --config scripts/subjective_randomness/configs/model_recovery.yaml \\
        --out data/subjective_randomness/model_recovery/confusion.json \\
        --tidy-csv data/subjective_randomness/model_recovery/confusion.csv
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
from src.subjective_randomness.model_recovery import (  # noqa: E402
    CONFUSION_COLUMNS,
    confusion_tidy_rows,
    run_recovery_from_config,
)
from src.subjective_randomness.tidy import write_tidy_csv  # noqa: E402


@dataclass
class Args:
    """Run closed-ended model recovery over the subjective-randomness seed set."""

    config: Path
    """YAML config (seed models, stimuli, generating params, fit settings)."""
    out: Path
    """Output JSON report path (the full confusion result)."""
    tidy_csv: Optional[Path] = None
    """Optional long-format CSV (one row per generating x recovered cell)."""
    results_root: Optional[Path] = None
    """Where to write per-model inner-loop artifacts (default: next to --out)."""
    n_participants: Optional[int] = None
    """Override the config's synthetic participants per generating model."""
    cache_dir: Optional[Path] = None
    """Optional directory for the PyMC .nc fit cache."""
    seed: Optional[int] = None
    """Override the config's RNG seed for the synthetic choices."""
    generator: Optional[Literal["pymc", "model_family"]] = None
    """Override the data source: the PyMC seed model, or the pure-Python family."""
    draws: Optional[int] = None
    """Override MCMC posterior draws per chain."""
    tune: Optional[int] = None
    """Override MCMC tuning (warmup) steps per chain."""
    chains: Optional[int] = None
    """Override the number of MCMC chains."""


def main(args: Args) -> None:
    config_path = resolve_path(args.config)
    out_path = resolve_path(args.out)
    results_root = (
        resolve_path(args.results_root)
        if args.results_root is not None
        else out_path.parent / f"{out_path.stem}_runs"
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

    result = run_recovery_from_config(
        load_config(config_path),
        config_path,
        results_root,
        n_participants_override=args.n_participants,
        fit_overrides=fit_overrides or None,
        seed_override=args.seed,
        generator_override=args.generator,
        cache_dir=resolve_path(args.cache_dir) if args.cache_dir else None,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    n_correct = sum(1 for r in result["generating"] if r["recovered_correct"])
    n_total = len(result["generating"])
    print(
        f"Wrote model-recovery confusion for {n_total} generating model(s) "
        f"({n_correct}/{n_total} recovered as best) to {out_path}"
    )

    if args.tidy_csv is not None:
        tidy_path = resolve_path(args.tidy_csv)
        write_tidy_csv(
            confusion_tidy_rows(result), tidy_path, columns=CONFUSION_COLUMNS
        )
        print(f"Wrote tidy confusion CSV to {tidy_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
