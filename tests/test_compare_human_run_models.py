"""Tests for the cross-run best-model pairwise-RMSE comparison script.

``compare_human_run_models`` takes the best-fitting model(s) each human run
arrived at (one per run, or both when two are tied in the posterior) and
measures how similar they are by the RMSE between their fitted ``p_left``
predictions over an exhaustive stimulus pool — the same prediction-and-RMSE
machinery the model-recovery analysis uses. These tests cover the pure helpers
(winner selection, the RMSE matrix); the MCMC fit/predict step is exercised by
running the script, not unit-tested here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts" / "analysis"


def _load_cli():
    """Load the analysis script as a module (its helpers are the unit under test)."""
    spec = importlib.util.spec_from_file_location(
        "compare_human_run_models", SCRIPTS / "compare_human_run_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


# --- select_winning_models -------------------------------------------------


def test_single_dominant_winner_selected_alone():
    post = {"a": 0.969, "b": 0.019, "c": 0.012}
    assert cli.select_winning_models(post, tie_ratio=0.5) == ["a"]


def test_two_near_equal_models_both_selected():
    # run2/experiment3: a near-perfect tie -> both models are winners.
    post = {"inner_loop_model": 0.492, "evidence_accumulation_per_run": 0.46, "z": 0.048}
    assert cli.select_winning_models(post, tie_ratio=0.5) == [
        "inner_loop_model",
        "evidence_accumulation_per_run",
    ]


def test_tie_ratio_excludes_models_below_threshold():
    post = {"x": 0.60, "y": 0.25}  # 0.25 < 0.5 * 0.60 = 0.30 -> excluded
    assert cli.select_winning_models(post, tie_ratio=0.5) == ["x"]


def test_winners_returned_in_descending_posterior_order():
    # All three within tie_ratio of the top; order is by descending posterior.
    post = {"low": 0.30, "high": 0.40, "mid": 0.30}
    assert cli.select_winning_models(post, tie_ratio=0.5) == ["high", "low", "mid"]


def test_empty_posteriors_raises():
    with pytest.raises(ValueError):
        cli.select_winning_models({}, tie_ratio=0.5)


def test_all_zero_posteriors_raises():
    with pytest.raises(ValueError):
        cli.select_winning_models({"a": 0.0, "b": 0.0}, tie_ratio=0.5)


# --- pairwise_rmse_matrix --------------------------------------------------


def test_rmse_matrix_zero_on_diagonal_and_symmetric():
    preds = {
        "m1": np.array([0.1, 0.2, 0.3]),
        "m2": np.array([0.1, 0.2, 0.3]),
        "m3": np.array([0.4, 0.2, 0.0]),
    }
    labels, mat = cli.pairwise_rmse_matrix(preds)
    assert labels == ["m1", "m2", "m3"]
    assert np.allclose(np.diag(mat), 0.0)
    assert np.allclose(mat, mat.T)
    assert mat[0, 1] == pytest.approx(0.0)  # m1 and m2 are identical


def test_rmse_value_matches_formula():
    preds = {"a": np.array([0.0, 0.0]), "b": np.array([0.3, 0.4])}
    _labels, mat = cli.pairwise_rmse_matrix(preds)
    # rmse = sqrt(mean([0.09, 0.16])) = sqrt(0.125)
    assert mat[0, 1] == pytest.approx(np.sqrt(0.125))


def test_rmse_matrix_unequal_lengths_raises():
    preds = {"a": np.array([0.1, 0.2]), "b": np.array([0.1])}
    with pytest.raises(ValueError):
        cli.pairwise_rmse_matrix(preds)


def test_rmse_matrix_needs_two_models():
    with pytest.raises(ValueError):
        cli.pairwise_rmse_matrix({"only": np.array([0.1, 0.2])})


# --- resolve_winners -------------------------------------------------------


def _write_posterior(path: Path, posteriors: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"posteriors": posteriors, "elpd_loo": {}, "comparison": {}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_winners_reads_each_run(tmp_path):
    exp = "experiment3"
    specs = {
        "run1": {"iter1_candidate0": 0.969, "other": 0.031},
        "run2": {
            "inner_loop_model": 0.492,
            "evidence_accumulation_per_run": 0.46,
            "z": 0.048,
        },
        "run3": {"iter1_candidate0": 0.994, "x": 0.006},
    }
    for run, post in specs.items():
        path = (
            tmp_path
            / run
            / "subjective_randomness"
            / exp
            / "model_loop"
            / "model_posterior.json"
        )
        _write_posterior(path, post)

    winners = cli.resolve_winners(tmp_path, exp, tie_ratio=0.5)
    assert winners == [
        ("run1", "iter1_candidate0"),
        ("run2", "inner_loop_model"),
        ("run2", "evidence_accumulation_per_run"),
        ("run3", "iter1_candidate0"),
    ]


def test_resolve_winners_no_runs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.resolve_winners(tmp_path, "experiment3", tie_ratio=0.5)


def test_model_label_disambiguates_same_name_across_runs():
    assert cli.model_label("run1", "iter1_candidate0") == "run1/iter1_candidate0"
    assert cli.model_label("run3", "iter1_candidate0") == "run3/iter1_candidate0"
