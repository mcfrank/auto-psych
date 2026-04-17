"""
Preprocess raw Game of 24 think-aloud experiment CSV into the pipeline's responses.csv format.

Usage:
    python3 preprocess_data.py --input PATH_TO_RAW_CSV [--output PATH]

Default output: <script_dir>/data/responses.csv
"""

import argparse
import csv
import pathlib
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw think-aloud Game of 24 CSV to pipeline responses.csv"
    )
    parser.add_argument("--input", required=True, help="Path to raw experiment CSV")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <script_dir>/data/responses.csv)",
    )
    return parser.parse_args()


def is_true(val):
    """Return True if val represents a truthy boolean (handles both string and bool)."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def is_nonempty(val):
    """Return True if val is a non-empty, non-NaN string."""
    if val is None:
        return False
    s = str(val).strip()
    return s != "" and s.lower() != "nan"


def main():
    args = parse_args()

    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    script_dir = pathlib.Path(__file__).parent
    if args.output is None:
        output_path = script_dir / "data" / "responses.csv"
    else:
        output_path = pathlib.Path(args.output)

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
    rows = [r for r in rows if r.get("trial_type", "").strip() == "GameOfN-audio-recording"]
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
    main()
