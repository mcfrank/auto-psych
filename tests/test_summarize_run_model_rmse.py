"""Tests for the pairwise-RMSE descriptive-statistics script.

``summarize_run_model_rmse`` reads the ``..._pairs.csv`` written by
``compare_human_run_models`` (one row per run-pair, with an ``rmse`` column) and
prints descriptive statistics over those RMSE values.
"""

from __future__ import annotations

import pytest

from tests.paths import ANALYSIS_SCRIPTS_DIR, load_script_module


@pytest.fixture(scope="module")
def cli():
    """The analysis script, loaded as a module — its helpers are the units."""
    return load_script_module(ANALYSIS_SCRIPTS_DIR / "summarize_run_model_rmse.py")


def test_rmse_statistics_are_correct(cli):
    stats = cli.rmse_statistics([0.0, 2.0, 4.0])
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx(2.0)  # sample stdev of [0, 2, 4]
    assert stats["min"] == pytest.approx(0.0)
    assert stats["median"] == pytest.approx(2.0)
    assert stats["max"] == pytest.approx(4.0)


def test_rmse_statistics_empty_fails_loudly(cli):
    with pytest.raises(ValueError):
        cli.rmse_statistics([])


def test_cli_reads_pairs_and_prints(cli, tmp_path, capsys):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "model_a,model_b,rmse\na,b,0.10\na,c,0.20\nb,c,0.30\n", encoding="utf-8"
    )
    cli.main(cli.Args(pairs_csv=csv_path))
    out = capsys.readouterr().out
    assert "mean" in out
    assert "0.2000" in out  # mean of 0.1, 0.2, 0.3


def test_cli_missing_file_fails_loudly(cli, tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.main(cli.Args(pairs_csv=tmp_path / "nope.csv"))


def test_cli_missing_column_fails_loudly(cli, tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("model_a,model_b,other\na,b,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rmse"):
        cli.main(cli.Args(pairs_csv=csv_path))
