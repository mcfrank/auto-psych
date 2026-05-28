"""Preprocess subjective_randomness responses.csv → numeric feature columns.

Takes an "old"-style responses.csv with raw H/T sequence strings:
    participant_id, trial_index, sequence_a, sequence_b, chose_left, chose_right, model

and writes a new CSV that keeps the raw columns AND adds numeric feature columns
derived from each sequence. The numeric columns are what theorist PyMC models
pull into `pm.Data` containers (one container per numeric column name).

Feature columns per sequence (`a` and `b`):
    n_<x>             total length
    h_<x>             head count
    p_<x>             head proportion (heads / length)
    alts_<x>          alternation count (transitions between H and T)
    p_alts_<x>        alternation proportion (alts / (length - 1))
    max_run_<x>       length of the longest constant run

The `chose_left` column is preserved (binary 0/1) for the observed response.

Usage (run from repo root or anywhere):
    python3 cc_pipeline/projects/subjective_randomness/preprocess_data.py \\
        --input-csv  cc_pipeline/projects/subjective_randomness/experiment1/data/responses.csv \\
        [--output-csv cc_pipeline/projects/subjective_randomness/data/responses.csv]
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import tyro
from pyprojroot import here


def sequence_features(seq: str, suffix: str) -> Dict[str, int]:
    """Numeric features derived from a single H/T sequence string."""
    s = seq.strip().upper()
    n = len(s)
    h = sum(1 for c in s if c == "H")
    alts = sum(1 for i in range(1, n) if s[i] != s[i - 1])
    # Longest run of consecutive identical characters
    max_run = 0
    cur = 0
    prev = ""
    for c in s:
        if c == prev:
            cur += 1
        else:
            cur = 1
            prev = c
        if cur > max_run:
            max_run = cur
    return {
        f"n_{suffix}": n,
        f"h_{suffix}": h,
        f"alts_{suffix}": alts,
        f"max_run_{suffix}": max_run,
    }


def sequence_features_float(seq: str, suffix: str, n: int, alts: int, h: int) -> Dict[str, float]:
    """Float features (proportions) derived alongside the integer features."""
    return {
        f"p_{suffix}": (h / n) if n > 0 else 0.0,
        f"p_alts_{suffix}": (alts / (n - 1)) if n > 1 else 0.0,
    }


def featurize_stimulus(sequence_a: str, sequence_b: str) -> Dict[str, float]:
    """Return the full feature-column dict for a single candidate stimulus pair.

    Keys match the pm.Data container names theorist PyMC models use:
    n_a, h_a, p_a, alts_a, p_alts_a, max_run_a and the _b counterparts. The
    design step uses this to convert raw (sequence_a, sequence_b) tuples into
    pm.Data inputs for EIG scoring.
    """
    feats_a = sequence_features(sequence_a, "a")
    feats_b = sequence_features(sequence_b, "b")
    floats_a = sequence_features_float(sequence_a, "a", feats_a["n_a"], feats_a["alts_a"], feats_a["h_a"])
    floats_b = sequence_features_float(sequence_b, "b", feats_b["n_b"], feats_b["alts_b"], feats_b["h_b"])
    return {**feats_a, **feats_b, **floats_a, **floats_b}


@dataclass
class Args:
    """Command-line arguments for preprocess_data."""
    input_csv: Path
    output_csv: Path = here() / "cc_pipeline" / "projects" / "subjective_randomness" / "data" / "responses.csv"


def main(args: Args) -> None:
    input_path = args.input_csv
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"Error: input CSV is empty: {input_path}", file=sys.stderr)
        sys.exit(1)

    required = {"sequence_a", "sequence_b", "chose_left"}
    missing = required - set(rows[0].keys())
    if missing:
        print(f"Error: input CSV missing columns {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    int_feature_cols = [
        "n_a", "h_a", "alts_a", "max_run_a",
        "n_b", "h_b", "alts_b", "max_run_b",
    ]
    float_feature_cols = ["p_a", "p_alts_a", "p_b", "p_alts_b"]
    passthrough = ["participant_id", "trial_index", "sequence_a", "sequence_b", "chose_left"]
    out_cols = passthrough + int_feature_cols + float_feature_cols

    n_in = len(rows)
    out_rows = []
    for r in rows:
        seq_a = r["sequence_a"]
        seq_b = r["sequence_b"]
        feats_a = sequence_features(seq_a, "a")
        feats_b = sequence_features(seq_b, "b")
        floats_a = sequence_features_float(seq_a, "a", feats_a["n_a"], feats_a["alts_a"], feats_a["h_a"])
        floats_b = sequence_features_float(seq_b, "b", feats_b["n_b"], feats_b["alts_b"], feats_b["h_b"])
        out_rows.append({
            **{k: r.get(k, "") for k in passthrough},
            **feats_a, **feats_b, **floats_a, **floats_b,
        })

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Rows input:  {n_in}")
    print(f"Rows output: {len(out_rows)}")
    print(f"Output cols: {out_cols}")
    print(f"Written to:  {output_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
