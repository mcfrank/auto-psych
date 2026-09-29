"""Fast tests for writing tidy CSV rows (`src/subjective_randomness/tidy.py`)."""

from __future__ import annotations

import csv

import pytest

from src.subjective_randomness.tidy import write_tidy_csv

COLUMNS = ["gt_model", "step", "pearson_r"]


def test_write_tidy_csv_rejects_row_missing_declared_column(tmp_path):
    rows = [{"gt_model": "m", "step": 0}]  # missing pearson_r
    with pytest.raises(KeyError, match="missing columns"):
        write_tidy_csv(rows, tmp_path / "tidy.csv", columns=COLUMNS)


def test_write_tidy_csv_roundtrips(tmp_path):
    rows = [
        {"gt_model": "m", "step": 0, "pearson_r": 0.5, "extra": "dropped"},
        {"gt_model": "m", "step": 1, "pearson_r": 0.75, "extra": "dropped"},
    ]
    out = tmp_path / "tidy.csv"

    write_tidy_csv(rows, out, columns=COLUMNS)

    with out.open(encoding="utf-8", newline="") as f:
        read_back = list(csv.DictReader(f))
    assert [list(row) for row in read_back] == [COLUMNS, COLUMNS]
    assert [row["pearson_r"] for row in read_back] == ["0.5", "0.75"]
