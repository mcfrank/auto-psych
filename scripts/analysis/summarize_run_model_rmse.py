"""CLI: print descriptive statistics for the cross-run pairwise RMSE values.

``compare_human_run_models.py`` writes a ``..._pairs.csv`` with one row per pair
of runs' best-fitting models and an ``rmse`` column (the RMSE between their
fitted ``p_left`` predictions). This script just reads that column and prints
descriptive statistics — count, mean, std, min, quartiles, median, max.

It fails loudly if the file or the ``rmse`` column is missing.

Usage:
    # Default: the experiment3 pairs file.
    uv run python scripts/analysis/summarize_run_model_rmse.py

    # Another pairs file / column.
    uv run python scripts/analysis/summarize_run_model_rmse.py \\
        --pairs-csv data/results/human_experiment/human_run_model_rmse_experiment2_pairs.csv
"""

from __future__ import annotations

import csv
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import resolve_path  # noqa: E402

DEFAULT_PAIRS_CSV = Path(
    "data/results/human_experiment/human_run_model_rmse_experiment3_pairs.csv"
)


@dataclass
class Args:
    """Print descriptive statistics for the cross-run pairwise RMSE values."""

    pairs_csv: Path = DEFAULT_PAIRS_CSV
    """The ``..._pairs.csv`` written by ``compare_human_run_models.py``."""
    column: str = "rmse"
    """Numeric column to summarize."""


def rmse_statistics(values: Sequence[float]) -> "OrderedDict[str, float]":
    """Descriptive statistics for a list of pairwise RMSE values."""
    if len(values) == 0:
        raise ValueError("no RMSE values to summarize")
    arr = np.asarray(values, dtype=float)
    return OrderedDict(
        [
            ("n", int(arr.size)),
            ("mean", float(arr.mean())),
            ("std", float(arr.std(ddof=1)) if arr.size > 1 else 0.0),
            ("min", float(arr.min())),
            ("25%", float(np.percentile(arr, 25))),
            ("median", float(np.median(arr))),
            ("75%", float(np.percentile(arr, 75))),
            ("max", float(arr.max())),
        ]
    )


def read_rmse_values(path: Path, column: str) -> list[float]:
    """Read the ``column`` of a pairs CSV as floats, failing loudly if absent."""
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            raise ValueError(
                f"no {column!r} column in {path} (have {reader.fieldnames})"
            )
        return [float(row[column]) for row in reader]


def main(args: Args) -> None:
    path = resolve_path(args.pairs_csv)
    if not path.exists():
        raise FileNotFoundError(f"No pairs CSV at {path}")

    stats = rmse_statistics(read_rmse_values(path, args.column))
    print(f"Descriptive statistics for {args.column!r} in {path}")
    for key, value in stats.items():
        if isinstance(value, int):
            print(f"  {key:>7}: {value}")
        else:
            print(f"  {key:>7}: {value:.4f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
