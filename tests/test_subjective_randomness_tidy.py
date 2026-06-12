"""Fast tests for reshaping parameter-recovery reports into tidy CSV rows.

Parameter fitting is Bayesian-only, so the sole supported report shape is the
PyMC one (`run["posterior"][param]["mean"]`). Runs carrying only the deleted
max-likelihood shape (`run["fit"]`) must be rejected loudly.
"""

from __future__ import annotations

import csv

import pytest

from src.subjective_randomness.tidy import (
    parameter_recovery_tidy_rows,
    write_tidy_csv,
)

# A minimal PyMC recovery report (as written by pymc_recover.py): each run
# carries a `posterior` whose per-parameter `mean` is that repeat's estimate.
PYMC_REPORT = {
    "model": "bayesian_diagnosticity",
    "true_params": {"alt_prior": 0.4, "beta": 4.0},
    "n_repeats": 2,
    "summary": {
        "alt_prior": {"true": 0.4, "mean_posterior_mean": 0.38},
        "beta": {"true": 4.0, "mean_posterior_mean": 4.2},
    },
    "runs": [
        {
            "repeat": 0,
            "posterior": {
                "alt_prior": {"mean": 0.33, "sd": 0.1},
                "beta": {"mean": 4.5, "sd": 0.5},
            },
        },
        {
            "repeat": 1,
            "posterior": {
                "alt_prior": {"mean": 0.43, "sd": 0.1},
                "beta": {"mean": 3.9, "sd": 0.5},
            },
        },
    ],
}


def test_pymc_report_yields_one_row_per_param_and_repeat():
    rows = parameter_recovery_tidy_rows(PYMC_REPORT)

    assert len(rows) == 4  # 2 params x 2 repeats
    alt_r0 = next(
        r for r in rows if r["parameter"] == "alt_prior" and r["repeat"] == 0
    )
    assert alt_r0["model"] == "bayesian_diagnosticity"
    assert alt_r0["true_value"] == 0.4
    assert alt_r0["estimate"] == 0.33  # posterior mean, not the summary mean
    assert alt_r0["error"] == pytest.approx(0.33 - 0.4)


# A sampled-truth report (the pymc_recover.py default): no top-level
# `true_params`; each run carries the ground-truth vector it was simulated from.
SAMPLED_REPORT = {
    "model": "prototype_similarity",
    "param_ranges": {"theta_alt": [0.05, 0.95], "beta": [0.2, 12.0]},
    "n_repeats": 2,
    "runs": [
        {
            "repeat": 0,
            "true_params": {"theta_alt": 0.2, "beta": 1.5},
            "posterior": {
                "theta_alt": {"mean": 0.25, "sd": 0.05},
                "beta": {"mean": 1.8, "sd": 0.4},
            },
        },
        {
            "repeat": 1,
            "true_params": {"theta_alt": 0.8, "beta": 9.0},
            "posterior": {
                "theta_alt": {"mean": 0.75, "sd": 0.05},
                "beta": {"mean": 8.0, "sd": 0.6},
            },
        },
    ],
}


def test_sampled_report_pairs_each_estimate_with_its_runs_truth():
    rows = parameter_recovery_tidy_rows(SAMPLED_REPORT)

    assert len(rows) == 4  # 2 params x 2 repeats
    theta_r0 = next(
        r for r in rows if r["parameter"] == "theta_alt" and r["repeat"] == 0
    )
    theta_r1 = next(
        r for r in rows if r["parameter"] == "theta_alt" and r["repeat"] == 1
    )
    # Each row uses its own run's ground truth, not a shared report-level one.
    assert theta_r0["true_value"] == 0.2
    assert theta_r1["true_value"] == 0.8
    assert theta_r0["error"] == pytest.approx(0.25 - 0.2)
    assert theta_r1["error"] == pytest.approx(0.75 - 0.8)


def test_tidy_rows_reject_report_without_any_truth():
    # Neither a top-level `true_params` nor per-run truths: fail loudly.
    malformed = {
        "model": "prototype_similarity",
        "runs": [{"repeat": 0, "posterior": {"theta_alt": {"mean": 0.5}}}],
    }
    with pytest.raises(KeyError, match="true_params"):
        parameter_recovery_tidy_rows(malformed)


def test_tidy_rows_reject_run_without_posterior():
    # Runs without a posterior — including the deleted max-likelihood
    # `run["fit"]` shape — must fail loudly, not silently skip the repeat or
    # quietly read a point estimate from a shape nothing produces anymore.
    legacy_ml = {
        "model": "prototype_similarity",
        "true_params": {"theta_alt": 0.65},
        "runs": [{"repeat": 0, "fit": {"params": {"theta_alt": 0.5}}}],
    }
    with pytest.raises(KeyError, match="posterior"):
        parameter_recovery_tidy_rows(legacy_ml)


def test_write_tidy_csv_rejects_row_missing_declared_column(tmp_path):
    rows = [{"model": "m", "parameter": "beta"}]  # missing repeat/true_value/...
    with pytest.raises(KeyError, match="missing columns"):
        write_tidy_csv(rows, tmp_path / "tidy.csv")


def test_write_tidy_csv_roundtrips(tmp_path):
    rows = parameter_recovery_tidy_rows(PYMC_REPORT)
    out = tmp_path / "tidy.csv"

    write_tidy_csv(rows, out)

    with out.open(encoding="utf-8", newline="") as f:
        read_back = list(csv.DictReader(f))
    assert len(read_back) == 4
    assert set(read_back[0]) == {
        "model",
        "parameter",
        "repeat",
        "true_value",
        "estimate",
        "error",
    }
