"""CLI: discriminating-stimulus probe for a finished holdout-recovery run.

Reads an existing holdout-recovery result JSON (the ``--out`` of
``holdout_recovery.py``) and, for every held-out ground truth, mines a large
in-distribution stimulus pool for the pairs where the recovered winning model
and the ground truth disagree most. It then contrasts agreement on that
adversarial set with a matched random control and the whole pool, writes a
summary JSON and a scatter figure, and prints a table.

This is the test the loop itself cannot run: it uses the held-out ground truth
(which the loop never sees) to find where the recovered model would break.

Usage:
    uv run python scripts/subjective_randomness/probe_holdout_recovery.py \\
        --result data/subjective_randomness/holdout_recovery_v2/holdout.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import resolve_path  # noqa: E402
from src.subjective_randomness.discriminating_probe import (  # noqa: E402
    plot_probe,
    probe_summary_text,
    run_probe,
)


@dataclass
class Args:
    """Discriminating-stimulus probe for a finished holdout-recovery result."""

    result: Path
    """Existing holdout-recovery result JSON to probe."""
    out: Optional[Path] = None
    """Summary JSON path (default: <result dir>/<stem>_probe.json)."""
    figure: Optional[Path] = None
    """Scatter figure path (default: <result dir>/<stem>_probe.png)."""
    cache_dir: Optional[Path] = None
    """PyMC .nc fit cache for the winner refits (default: <result dir>/probe_cache)."""
    pool_size: int = 8000
    """Candidate pool size to mine for disagreement."""
    top_k: int = 300
    """Number of max-disagreement stimuli in the adversarial set."""
    pool_seed: int = 101
    """Seed for the candidate pool."""
    control_seed: int = 202
    """Seed for the matched random control subset."""
    draws: int = 1000
    """MCMC draws per chain for the winner refits. The probe only needs a stable
    posterior-mean p_left, so this is lighter than the original run's fit by
    default."""
    tune: int = 1000
    """MCMC tuning steps per chain for the winner refits."""
    chains: int = 2
    """MCMC chains for the winner refits."""


def _strip_plot_keys(obj: Any) -> Any:
    """Drop the heavy ``_plot`` arrays before writing the summary JSON."""
    if isinstance(obj, dict):
        return {k: _strip_plot_keys(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_plot_keys(v) for v in obj]
    return obj


def main(args: Args) -> None:
    result_path = resolve_path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    cache_dir = (
        resolve_path(args.cache_dir)
        if args.cache_dir is not None
        else result_path.parent / "probe_cache"
    )

    probe = run_probe(
        result,
        cache_dir=cache_dir,
        pool_size=args.pool_size,
        k=args.top_k,
        pool_seed=args.pool_seed,
        control_seed=args.control_seed,
        fit_kwargs={"draws": args.draws, "tune": args.tune, "chains": args.chains},
    )

    print(probe_summary_text(probe))

    out_path = (
        resolve_path(args.out)
        if args.out is not None
        else result_path.with_name(f"{result_path.stem}_probe.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_strip_plot_keys(probe), indent=2), encoding="utf-8")
    print(f"\nWrote probe summary to {out_path}")

    figure_path = (
        resolve_path(args.figure)
        if args.figure is not None
        else result_path.with_name(f"{result_path.stem}_probe.png")
    )
    plot_probe(probe, figure_path)
    print(f"Wrote probe figure to {figure_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
