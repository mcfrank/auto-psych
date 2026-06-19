"""Unit tests for the CriticAL posterior-predictive critique core.

The critique step generates posterior-predictive replicates from a *fitted*
PyMC model and compares an LLM-proposed test statistic computed on the observed
responses against the distribution of that statistic over the replicates. These
tests cover the pure statistical machinery (empirical p-values, test-statistic
file parsing) and the frame builder that swaps the
observed-response column for each posterior-predictive draw. The frame-builder
test stubs ``sample_synthetic_responses`` so it never runs MCMC.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.critique.ppc import (
    TestStatistic,
    build_critique_frames,
    evaluate_test_statistic,
    evaluate_test_stat_dir,
    load_test_statistic_file,
    run_ppc_for_model,
)
from src.models.pymc_inference import FittedModel, load_pymc_model

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pymc_models"

MEAN_RESPONSE_CODE = (
    "# name: mean_response\n"
    "# description: Mean of the observed binary response.\n"
    "def test_statistic(df):\n"
    "    return float(df['chose_left'].mean())\n"
)


def _frames(observed, replicates):
    """Build a human frame and model replicate frames from raw response arrays.

    ``observed`` is the observed-response column; ``replicates`` is a list of
    synthetic-response columns. Every frame carries one feature column so test
    statistics that condition on features have something to read.
    """
    n = len(observed)
    feature = list(range(n))
    human = pd.DataFrame({"chose_left": observed, "n_a": feature})
    models = [
        pd.DataFrame({"chose_left": rep, "n_a": feature}) for rep in replicates
    ]
    return human, models


def test_evaluate_test_statistic_two_sided_pvalue_and_zscore():
    ts = TestStatistic(
        name="mean_response", code=MEAN_RESPONSE_CODE, description="mean"
    )
    # Observed mean = 1.0; null means spread 0.0..0.8 → observed is at/above all,
    # so the two-sided p-value hits the plus-one Monte-Carlo floor and z > 0.
    human, models = _frames(
        observed=[1, 1, 1, 1],
        replicates=[[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]],
    )
    res = evaluate_test_statistic(ts, human, models)

    assert res.error is None
    assert res.t_observed == 1.0
    assert len(res.t_null) == 4
    assert res.z_score > 0
    assert res.p_value_is_floor is True
    # n_ge counts nulls >= 1.0 → only the all-zeros excluded? none equal 1.0 → 0.
    # Two-sided p = 2 * min((n_ge+1)/(n+1), (n_le+1)/(n+1)).
    n = 4
    null_means = [0.0, 0.25, 0.5, 0.75]
    n_ge = sum(1 for m in null_means if m >= 1.0)
    n_le = sum(1 for m in null_means if m <= 1.0)
    expected = min(1.0, 2.0 * min((n_ge + 1) / (n + 1), (n_le + 1) / (n + 1)))
    assert math.isclose(res.p_value, expected)


def test_evaluate_test_statistic_records_execution_error():
    ts = TestStatistic(
        name="boom",
        code="def test_statistic(df):\n    raise RuntimeError('nope')\n",
        description="raises",
    )
    human, models = _frames([1, 0], [[0, 0], [1, 1]])
    res = evaluate_test_statistic(ts, human, models)

    assert res.error is not None
    assert "RuntimeError" in res.error
    assert math.isnan(res.p_value)


def test_load_test_statistic_file_parses_name_and_description(tmp_path):
    path = tmp_path / "my_stat.py"
    path.write_text(MEAN_RESPONSE_CODE, encoding="utf-8")
    ts = load_test_statistic_file(path)
    assert ts.name == "mean_response"
    assert ts.description == "Mean of the observed binary response."
    assert "def test_statistic" in ts.code


def test_load_test_statistic_file_defaults_name_to_stem(tmp_path):
    path = tmp_path / "fallback_name.py"
    path.write_text(
        "def test_statistic(df):\n    return float(df['chose_left'].mean())\n",
        encoding="utf-8",
    )
    ts = load_test_statistic_file(path)
    assert ts.name == "fallback_name"


def _fitted_with_stub_ppc(synthetic: np.ndarray) -> FittedModel:
    """A FittedModel on the fixture graph whose PPC returns a fixed array.

    No MCMC: the real model graph supports ``make_stim_data`` /
    ``observed_response_data``; ``sample_synthetic_responses`` is replaced with a
    deterministic stub so the frame builder is exercised without sampling.
    """
    model = load_pymc_model("bayesian_fair_coin", FIXTURE_DIR)
    fitted = FittedModel(name="bayesian_fair_coin", model=model, idata=None, fingerprint="x")
    fitted.sample_synthetic_responses = (  # type: ignore[method-assign]
        lambda stim_data, n_datasets, seed=42: np.asarray(synthetic)[:n_datasets]
    )
    return fitted


def test_build_critique_frames_swaps_response_for_each_replicate():
    df = pd.read_csv(FIXTURE_DIR / "responses.csv")
    n = len(df)
    # Two synthetic datasets: all-zeros and all-ones responses.
    synthetic = np.vstack([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    fitted = _fitted_with_stub_ppc(synthetic)

    human_df, model_dfs = build_critique_frames(
        fitted, FIXTURE_DIR / "responses.csv", n_replicates=2, seed=0
    )

    # Human frame holds the real observed responses and all feature columns.
    assert list(human_df["chose_left"]) == list(df["chose_left"])
    for col in ("n_a", "h_a", "n_b", "h_b"):
        assert list(human_df[col]) == list(df[col])

    # Each replicate frame keeps the features but uses the synthetic response.
    assert len(model_dfs) == 2
    assert list(model_dfs[0]["chose_left"]) == [0] * n
    assert list(model_dfs[1]["chose_left"]) == [1] * n
    assert list(model_dfs[0]["n_a"]) == list(df["n_a"])


def test_evaluate_test_stat_dir_marks_significant(tmp_path, monkeypatch):
    df = pd.read_csv(FIXTURE_DIR / "responses.csv")
    n = len(df)
    # All replicates predict every response is 0; observed mean is well above 0,
    # so the mean-response statistic is a strong, significant discrepancy. 200
    # replicates put the two-sided floor (2/201) below α even after BH (m=2).
    synthetic = np.zeros((200, n), dtype=int)
    fitted = _fitted_with_stub_ppc(synthetic)

    stats_dir = tmp_path / "test_stats"
    stats_dir.mkdir()
    (stats_dir / "mean_response.py").write_text(MEAN_RESPONSE_CODE, encoding="utf-8")
    (stats_dir / "constant.py").write_text(
        "# name: constant\n"
        "# description: A constant — never a discrepancy.\n"
        "def test_statistic(df):\n    return 0.0\n",
        encoding="utf-8",
    )

    out = evaluate_test_stat_dir(
        fitted,
        FIXTURE_DIR / "responses.csv",
        stats_dir,
        n_replicates=200,
        seed=0,
        significance_alpha=0.05,
    )

    results = {r["name"]: r for r in out["results"]}
    assert results["mean_response"]["significant"] is True
    assert results["mean_response"]["p_value"] <= 0.05
    # A statistic identical on observed and replicates is never significant.
    assert results["constant"]["significant"] is False
    assert out["n_significant"] == 1


@pytest.mark.slow
def test_run_ppc_for_model_end_to_end_real_mcmc(tmp_path):
    """Fit the fixture model for real and run the PPC over real posterior draws.

    Exercises the full path (MCMC fit → posterior-predictive replicates →
    empirical p-values), which the stubbed tests above do not. A statistic the
    well-fit model reproduces should not be a strong discrepancy.
    """
    stats_dir = tmp_path / "test_stats"
    stats_dir.mkdir()
    (stats_dir / "mean_response.py").write_text(MEAN_RESPONSE_CODE, encoding="utf-8")

    out = run_ppc_for_model(
        "bayesian_fair_coin",
        FIXTURE_DIR,
        FIXTURE_DIR / "responses.csv",
        stats_dir,
        cache_dir=tmp_path / "cache",
        n_replicates=200,
        significance_alpha=0.05,
        fit_kwargs={"draws": 300, "tune": 300, "chains": 2},
    )

    res = out["results"][0]
    assert res["name"] == "mean_response"
    assert res["n_replicates"] == 200
    assert math.isfinite(res["t_observed"])
    assert math.isfinite(res["p_value"])
    # The data were simulated from this model, so its own PPC reproduces the
    # mean response: the observed value sits inside the replicate distribution.
    assert res["significant"] is False
