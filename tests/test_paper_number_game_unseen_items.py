"""Unit tests for ``scripts/paper/number_game_unseen_items.py`` (no fitting):
which responses are held out, their predictive density, and the scoring
against the best seed."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.paper import number_game_unseen_items as un


def _row(number_set, target, rating=0):
    return {"number_set": number_set, "target": target, "trial_index": 0, "rating": rating}


def test_held_out_responses_are_the_undrawn_items_and_flag_undrawn_sets():
    drawn = [_row("2,4", "6")]
    everything = [_row("2,4", "6"), _row("2,4", "8"), _row("3,9", "27"), _row("3,9", "27")]
    rows, set_unseen = un.unseen_responses(everything, drawn)
    assert [(r["number_set"], r["target"]) for r in rows] == [("2,4", "8"), ("3,9", "27"), ("3,9", "27")]
    assert set_unseen.tolist() == [False, True, True]


def test_pointwise_lpd_averages_the_probability_over_draws_before_the_log():
    draws = np.array([[0.9, 0.2], [0.5, 0.6]])
    lpd = un.pointwise_lpd(draws, np.array([1, 0]))
    assert lpd == pytest.approx([np.log(0.7), np.log(0.6)])


def test_models_are_scored_against_the_seed_best_on_these_rows():
    rows = [_row("a", "1", 1), _row("a", "2", 0), _row("b", "1", 1), _row("b", "2", 0)]
    lpd = {"win": np.array([-0.1, -0.1, -0.2, -0.2]),
           "s1": np.array([-1.0, -1.0, -1.0, -1.0]),
           "s2": np.array([-0.5, -0.5, -0.5, -0.6])}
    p_yes = {n: np.array([0.9, 0.1, 0.7, 0.4]) for n in lpd}
    scored = un.score(rows, lpd, p_yes, seeds=["s1", "s2"])
    assert scored["best_seed"] == "s2"
    assert scored["n_example_sets"] == 2
    win = scored["models"]["win"]["vs_best_seed"]
    assert win["diff"] == pytest.approx(1.5)
    # Example-set sums of the differences are 0.8 and 0.7: sqrt(2 * var).
    assert win["se"] == pytest.approx(np.sqrt(2 * np.var([0.8, 0.7])))
    assert scored["models"]["s2"]["vs_best_seed"] == {"diff": 0.0, "se": 0.0}
