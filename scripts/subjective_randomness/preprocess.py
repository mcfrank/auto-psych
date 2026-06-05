"""CLI: featurize a raw subjective_randomness responses.csv.

Usage:
    uv run python scripts/subjective_randomness/preprocess.py \\
        --input-csv  data/subjective_randomness/experiment1/responses.csv \\
        [--output-csv data/subjective_randomness/responses.csv]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.features import featurize_responses_csv  # noqa: E402

DEFAULT_OUTPUT = here() / "data" / "subjective_randomness" / "responses.csv"


@dataclass
class Args:
    """Featurize a raw responses.csv into numeric model-input columns."""

    input_csv: Path
    """Raw responses.csv with sequence_a / sequence_b / chose_left columns."""
    output_csv: Path = DEFAULT_OUTPUT
    """Where to write the featurized CSV."""


def main(args: Args) -> None:
    n_written = featurize_responses_csv(args.input_csv, args.output_csv)
    print(f"Wrote {n_written} featurized rows to {args.output_csv}")


if __name__ == "__main__":
    main(tyro.cli(Args))
