"""Unit tests for the discriminating-stimulus probe's pure helpers.

The fit/predict orchestration in ``probe_gt_run`` needs a real cached run and
PyMC, so it is exercised by running the CLI on an actual holdout result, not
here. These tests pin the deterministic selection and scoring logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.subjective_randomness.discriminating_probe import (
    _agreement_scores,
    select_by_disagreement,
)


def test_select_by_disagreement_picks_largest_gap():
    gt = [0.5, 0.5, 0.5, 0.5]
    winner = [0.5, 0.9, 0.1, 0.55]  # |Δ| = 0.0, 0.4, 0.4, 0.05
    assert select_by_disagreement(gt, winner, 2) == [1, 2]  # ties break by index


def test_select_by_disagreement_orders_descending():
    gt = [0.0, 0.0, 0.0]
    winner = [0.1, 0.9, 0.4]  # |Δ| = 0.1, 0.9, 0.4
    assert select_by_disagreement(gt, winner, 3) == [1, 2, 0]


def test_select_by_disagreement_rejects_bad_k():
    with pytest.raises(ValueError, match="k must be in"):
        select_by_disagreement([0.1], [0.2], 2)
    with pytest.raises(ValueError, match="k must be in"):
        select_by_disagreement([0.1, 0.2], [0.2, 0.3], 0)


def test_select_by_disagreement_rejects_length_mismatch():
    with pytest.raises(ValueError, match="differ in length"):
        select_by_disagreement([0.1, 0.2], [0.2], 1)


def test_agreement_scores_on_subset():
    gt = np.array([0.2, 0.8, 0.5, 0.9])
    winner = np.array([0.2, 0.8, 0.5, 0.4])  # only index 3 disagrees (0.5)
    scores = _agreement_scores(gt, winner, np.array([3]))
    assert scores["n"] == 1
    assert scores["max_abs_disagreement"] == pytest.approx(0.5)
    assert scores["mean_abs_disagreement"] == pytest.approx(0.5)
    assert scores["rmse"] == pytest.approx(0.5)


def test_agreement_scores_perfect_match_full_set():
    gt = np.array([0.2, 0.8, 0.5])
    scores = _agreement_scores(gt, gt.copy(), np.arange(3))
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["max_abs_disagreement"] == pytest.approx(0.0)
    assert scores["pearson_r"] == pytest.approx(1.0)
