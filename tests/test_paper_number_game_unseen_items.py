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


# --- the non-winning models (Figure 2's strip) --------------------------------------------


def test_the_strip_is_the_pooled_non_winning_models():
    pooled = {"pruned_during_run": {"r2__y": 1.0}, "others_at_end": {"r1__x": 2.0}}
    assert un.strip_labels(pooled) == ["r1__x", "r2__y"]


def test_a_strip_record_sums_each_test_sets_responses():
    record = un.strip_record(np.array([-1.0, -2.0, -3.0]), np.array([True, False, True]))
    assert record == {"unseen_items": -6.0, "unseen_example_sets": -4.0}


def test_each_strip_model_is_drawn_against_its_test_sets_best_seed(tmp_path):
    for label, sums in {"r1__a": (-12.0, -3.0), "r2__b": (-9.0, -5.0)}.items():
        un.save_strip_record(tmp_path, label, {"unseen_items": sums[0], "unseen_example_sets": sums[1]})
    record = {"unseen_items": {"best_seed": "s", "models": {"s": {"lpd": -10.0}}},
              "unseen_example_sets": {"best_seed": "t", "models": {"t": {"lpd": -4.0}}}}
    out = un.with_strip(record, un.collect_strip(["r1__a", "r2__b"], tmp_path))
    assert out["unseen_items"]["strip"] == {"r1__a": -2.0, "r2__b": 1.0}
    assert out["unseen_example_sets"]["strip"] == {"r1__a": 1.0, "r2__b": -1.0}


def test_a_strip_model_without_a_record_stops_the_assembly(tmp_path):
    un.save_strip_record(tmp_path, "r1__a", {"unseen_items": -1.0, "unseen_example_sets": -1.0})
    with pytest.raises(FileNotFoundError, match="r2__b"):
        un.collect_strip(["r1__a", "r2__b"], tmp_path)
