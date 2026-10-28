"""Unit tests for ``scripts/paper/number_game_figure_data.py`` (no fitting):
the scoring against the best seed with a stand-in comparison, the condition
table, and what it reads from the committed dataset run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.paper import number_game_figure_data as ng


def _fake_compare(elpd, se_between, not_converged=()):
    """compare_table restricted to ``names``: the lower model's row carries the
    clustered SE of its difference from the higher one."""

    def compare(responses, models_dir, *, names=None, **kwargs):
        top = max(names, key=elpd.get)
        return {
            n: {
                "elpd_loo": elpd[n],
                "dse_clustered": 0.0 if n == top else se_between.get(frozenset((n, top)), float("nan")),
                "loo_unreliable": False,
                "not_converged": n in not_converged,
            }
            for n in names
        }

    return compare


ELPD = {"win": -100.0, "other": -104.0, "s1": -150.0, "s2": -140.0}


def test_the_winner_and_others_are_scored_against_the_best_seed():
    compare = _fake_compare(ELPD, {frozenset(("win", "s2")): 6.0, frozenset(("s1", "s2")): 3.0})

    scored = ng.score_run("r.csv", "m", winner="win", others=["other"], seeds=["s1", "s2"],
                          compare=compare)

    assert scored["best_seed"] == "s2"
    assert scored["vs_best_seed"] == {
        "win": {"diff": 40.0, "se": 6.0},
        "s1": {"diff": -10.0, "se": 3.0},
        "s2": {"diff": 0.0, "se": 0.0},
    }
    assert scored["others_at_end"] == {"other": 36.0}
    assert scored["elpd"]["other"] == -104.0


def test_an_unconverged_other_live_model_stops_the_scoring():
    compare = _fake_compare(ELPD, {frozenset(("win", "s2")): 6.0, frozenset(("s1", "s2")): 3.0},
                            not_converged={"other"})

    with pytest.raises(RuntimeError, match="other: not converged"):
        ng.score_run("r.csv", "m", winner="win", others=["other"], seeds=["s1", "s2"],
                     compare=compare)


def _row(number_set, target, rating):
    return {"number_set": number_set, "target": str(target), "rating": str(rating)}


def test_conditions_are_example_set_and_target_cells():
    rows = [_row("2,4", 6, 1), _row("2,4", 6, 0), _row("2,4", 7, 0), _row("10,20", 6, 1)]
    p = np.array([0.8, 0.6, 0.1, 0.5])

    conditions = ng.condition_table(rows, {"m": p})

    by_cell = {(c["number_set"], c["target"]): c for c in conditions}
    assert by_cell[("2,4", 6)] == {"number_set": "2,4", "target": 6, "n": 2, "observed": 0.5,
                                   "m": pytest.approx(0.7)}
    assert by_cell[("2,4", 7)]["observed"] == 0.0
    assert by_cell[("10,20", 6)]["n"] == 1


def test_conditions_refuse_predictions_that_do_not_cover_every_response():
    with pytest.raises(ValueError, match="2 predictions for 1 responses"):
        ng.condition_table([_row("2,4", 6, 1)], {"m": np.array([0.5, 0.5])})


def test_the_committed_run_gives_its_settings_live_models_and_seed_files():
    settings = ng.run_settings(json.loads(ng.DEFAULT_HOLDOUT.read_text()))
    assert settings["fit_kwargs"] == {"method": "warm_start_nuts", "tune": 200, "draws": 500,
                                      "chains": 4}
    assert settings["selection"].as_record() == {"criterion": "grouped_cv",
                                                 "cv_group": ["number_set"],
                                                 "cv_folds": 4, "cv_seed": 0}

    final_dir = ng.sr.final_experiment_dir(ng.DEFAULT_RUN_DIR)
    live = ng.live_models(final_dir)
    assert final_dir.name == "experiment3"
    assert live["winner"] == "even_rule_digit_size_principle_counting"
    assert len(live["others"]) == 3
    assert live["n_trials"] == 7560
    # Figure 2 draws them in admission order with the pruned models.
    assert len(ng.rm.pruned_order(final_dir, live["others"])) == 3

    seeds = json.loads((ng.DEFAULT_RUN_DIR / "starting_models.json").read_text())["starting_models"]
    for seed in seeds:  # pruned in experiment 1: found in its pruned/ directory
        assert ng.sr.seed_file(ng.DEFAULT_RUN_DIR, final_dir, seed).exists()


def test_the_removed_models_are_every_admitted_model_not_live_at_the_end_nor_a_seed():
    final_dir = ng.sr.final_experiment_dir(ng.DEFAULT_RUN_DIR)
    live = ng.live_models(final_dir)
    seeds = json.loads((ng.DEFAULT_RUN_DIR / "starting_models.json").read_text())["starting_models"]

    removed = ng.rm.removed_models(ng.DEFAULT_RUN_DIR, final_dir,
                                   live=[live["winner"], *live["others"]], seeds=seeds)

    assert len(removed) == 84
    assert not set(removed) & {live["winner"], *live["others"], *seeds}
    # Dropped as unfittable, never moved to pruned/: found in models/.
    assert removed["bayesian_digit_feature_concepts"]["outcome"] == "dropped"
    assert sum(r["outcome"] == "pruned" for r in removed.values()) == 83
    assert all(r["file"].exists() for r in removed.values())


def test_the_models_pruned_at_the_end_of_experiment_3_keep_the_loops_elpd():
    final_dir = ng.sr.final_experiment_dir(ng.DEFAULT_RUN_DIR)
    live = ng.live_models(final_dir)
    seeds = json.loads((ng.DEFAULT_RUN_DIR / "starting_models.json").read_text())["starting_models"]
    removed = ng.rm.removed_models(ng.DEFAULT_RUN_DIR, final_dir,
                                   live=[live["winner"], *live["others"]], seeds=seeds)

    recorded = ng.rm.recorded_scores(final_dir, removed)

    assert len(recorded) == 32
    # A refit on the final data at the run's settings gave -4689.3915 (2026-10-19).
    assert recorded["digit_rule_size_principle_counting"] == pytest.approx(-4689.3915, abs=1e-4)
    # Exactly the models pruned (not retired) at the end of experiment 3.
    late = {n for n, r in removed.items() if r["outcome"] == "pruned"
            and r["context"] == "experiment3 end of experiment"
            and ng.rm.RETIRED_MARK not in r["detail"]}
    assert set(recorded) == late
    assert "bayesian_digit_feature_concepts" not in recorded  # dropped in experiment 3
    assert any(r["context"].startswith("experiment1") for n, r in removed.items()
               if n not in recorded)
