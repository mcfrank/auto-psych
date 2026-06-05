"""
Preprocess raw Game of 24 think-aloud experiment CSV into the pipeline's responses.csv format.

Usage:
    uv run python scripts/think_aloud_game24/preprocess.py --input-csv PATH_TO_RAW_CSV [--output-csv PATH]

Default output: data/think_aloud_game24/responses.csv
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here


def is_true(val: object) -> bool:
    """Return True if val represents a truthy boolean (handles both string and bool)."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def is_nonempty(val: object) -> bool:
    """Return True if val is a non-empty, non-NaN string."""
    if val is None:
        return False
    s = str(val).strip()
    return s != "" and s.lower() != "nan"


@dataclass
class Args:
    """Command-line arguments for preprocess_data."""

    input_csv: Path
    output_csv: Path = here() / "data" / "think_aloud_game24" / "responses.csv"


def main(args: Args) -> None:
    input_path = args.input_csv
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output_csv

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_cols = [
        "participant_id",
        "trial_index",
        "choices",
        "target",
        "correct",
        "lm_code_translation",
    ]

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n_input = len(rows)
    print(f"Rows input: {n_input}")

    # Filter 1: trial_type == "GameOfN-audio-recording"
    rows = [
        r for r in rows if r.get("trial_type", "").strip() == "GameOfN-audio-recording"
    ]
    print(f"Rows after trial_type filter: {len(rows)}")

    # Filter 2: practice is not True (handle both string and bool)
    rows = [r for r in rows if not is_true(r.get("practice", ""))]
    print(f"Rows after practice filter: {len(rows)}")

    # Filter 3: relevant == "1" or relevant == 1
    rows = [r for r in rows if str(r.get("relevant", "")).strip() in ("1", "1.0")]
    print(f"Rows after relevant filter: {len(rows)}")

    # Filter 4: lm_code_translation is not empty/NaN
    rows = [r for r in rows if is_nonempty(r.get("lm_code_translation", ""))]
    print(f"Rows after lm_code_translation filter: {len(rows)}")

    n_output = len(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "participant_id": row.get("pid", ""),
                    "trial_index": row.get("trial_index", ""),
                    "choices": row.get("choices", ""),
                    "target": row.get("target", ""),
                    "correct": row.get("correct", ""),
                    "lm_code_translation": row.get("lm_code_translation", ""),
                }
            )

    print(f"Rows output: {n_output}")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
