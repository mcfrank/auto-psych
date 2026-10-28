"""Unit tests for ``scripts/paper/number_game_figure_data.py`` (no fitting):
the scoring against the best seed with a stand-in comparison, the condition
table, and what it reads from the committed dataset run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.paper import number_game_figure_data as ng

RUN1_DIR = ng.run_dir(ng.DEFAULT_RUNS_ROOT, "run1")


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
    settings = ng.run_settings(json.loads((ng.DEFAULT_HARNESS_ROOT / "run1" / "holdout.json").read_text()))
    assert settings["fit_kwargs"] == {"method": "warm_start_nuts", "tune": 200, "draws": 500,
                                      "chains": 4}
    assert settings["selection"].as_record() == {"criterion": "grouped_cv",
                                                 "cv_group": ["number_set"],
                                                 "cv_folds": 4, "cv_seed": 0}

    final_dir = ng.sr.final_experiment_dir(RUN1_DIR)
    live = ng.live_models(final_dir)
    assert final_dir.name == "experiment3"
    assert live["winner"] == "even_rule_digit_size_principle_counting"
    assert len(live["others"]) == 3
    assert live["n_trials"] == 7560
    # Figure 2 draws them in admission order with the pruned models.
    assert len(ng.rm.pruned_order(final_dir, live["others"])) == 3

    seeds = json.loads((RUN1_DIR / "starting_models.json").read_text())["starting_models"]
    for seed in seeds:  # pruned in experiment 1: found in its pruned/ directory
        assert ng.sr.seed_file(RUN1_DIR, final_dir, seed).exists()


def test_the_removed_models_are_every_admitted_model_not_live_at_the_end_nor_a_seed():
    final_dir = ng.sr.final_experiment_dir(RUN1_DIR)
    live = ng.live_models(final_dir)
    seeds = json.loads((RUN1_DIR / "starting_models.json").read_text())["starting_models"]

    removed = ng.rm.removed_models(RUN1_DIR, final_dir,
                                   live=[live["winner"], *live["others"]], seeds=seeds)

    assert len(removed) == 84
    assert not set(removed) & {live["winner"], *live["others"], *seeds}
    # Dropped as unfittable, never moved to pruned/: found in models/.
    assert removed["bayesian_digit_feature_concepts"]["outcome"] == "dropped"
    assert sum(r["outcome"] == "pruned" for r in removed.values()) == 83
    assert all(r["file"].exists() for r in removed.values())


def test_the_models_pruned_at_the_end_of_experiment_3_keep_the_loops_elpd():
    final_dir = ng.sr.final_experiment_dir(RUN1_DIR)
    live = ng.live_models(final_dir)
    seeds = json.loads((RUN1_DIR / "starting_models.json").read_text())["starting_models"]
    removed = ng.rm.removed_models(RUN1_DIR, final_dir,
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


def _write_csv(path, header, rows):
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def test_pooled_responses_are_every_runs_rows_in_run_order(tmp_path):
    header = "number_set,target,trial_index,rating"
    one = _write_csv(tmp_path / "one.csv", header, ['"2,4",6,0,1'])
    two = _write_csv(tmp_path / "two.csv", header, ['"3,9",27,5,0', "10,20,7,1"])

    ng.pool_responses({"run1": one, "run2": two}, tmp_path / "pooled.csv")

    assert (tmp_path / "pooled.csv").read_text() == (
        header + '\n"2,4",6,0,1\n"3,9",27,5,0\n10,20,7,1\n')


def test_pooling_refuses_runs_with_different_columns(tmp_path):
    one = _write_csv(tmp_path / "one.csv", "number_set,target,trial_index,rating", [])
    two = _write_csv(tmp_path / "two.csv", "number_set,target,rating", [])

    with pytest.raises(RuntimeError, match="run2: .* has columns"):
        ng.pool_responses({"run1": one, "run2": two}, tmp_path / "pooled.csv")


def test_pooling_refuses_participant_ids_that_would_merge_people_across_runs(tmp_path):
    header = "number_set,target,participant_id,trial_index,rating"
    one = _write_csv(tmp_path / "one.csv", header, ['"2,4",6,0,0,1'])

    with pytest.raises(RuntimeError, match="participant_id"):
        ng.pool_responses({"run1": one}, tmp_path / "pooled.csv")


def test_runs_with_different_settings_are_not_pooled(tmp_path):
    holdout = json.loads((ng.DEFAULT_HARNESS_ROOT / "run1" / "holdout.json").read_text())
    for run, draws in (("run1", 500), ("run2", 1000)):
        (tmp_path / run).mkdir()
        changed = {**holdout, "fit_kwargs": {**holdout["fit_kwargs"], "draws": draws}}
        (tmp_path / run / "holdout.json").write_text(json.dumps(changed))

    with pytest.raises(RuntimeError, match="run2 .* differ"):
        ng.shared_run_settings(tmp_path, ["run1", "run2"])


RUNS = ["run1", "run2", "run3"]
WINNERS = {"run1": "even_rule_digit_size_principle_counting",
           "run2": "relational_analogy_transfer", "run3": "outlier_tolerant_rule_concepts"}
SEEDS = ["bigelow_piantadosi_independent", "bigelow_piantadosi_interval",
         "tenenbaum_rules_similarity"]


def _stand_in_elpd(name):
    """A fixed ELPD per model: winners above the seeds, seeds apart."""
    if name in SEEDS:
        return -5000.0 - 10 * SEEDS.index(name)
    if name in WINNERS.values():
        return -4600.0 - 10 * list(WINNERS.values()).index(name)
    return -4800.0 - (sum(map(ord, name)) % 97)


def test_the_figure_data_pool_all_three_runs(tmp_path, monkeypatch):
    """End to end on the committed runs, every fit stood in for: the shards
    score each run's removed models on its own data and every run's other
    models on all runs pooled, and the assembly writes each run's record and
    the pooled one Figures 2 and 3 draw."""
    import src.model_comparison.posterior as posterior
    import src.models.pymc_inference as pymc_inference

    def compare(responses, models_dir, *, names=None, **kwargs):
        top = max(names, key=_stand_in_elpd)
        return {n: {"elpd_loo": _stand_in_elpd(n), "dse_clustered": 0.0 if n == top else 5.0,
                    "loo_unreliable": False, "not_converged": False} for n in names}

    monkeypatch.setattr(posterior, "compare_table", compare)
    monkeypatch.setattr(pymc_inference, "fit_models_to_cache", lambda names, *a, **k: {})
    monkeypatch.setattr(pymc_inference, "clear_fit_cache", lambda: None)
    monkeypatch.setattr(ng, "score_removed_model",
                        lambda name, *a, **k: {"elpd": _stand_in_elpd(name)})
    monkeypatch.setattr(ng, "posterior_mean_p_yes",
                        lambda name, models_dir, responses, **k:
                        np.full(len(ng.sr._read_rows(responses)), 0.5))
    args = ng.Args(work_dir=tmp_path / "work", out=tmp_path / "number_game.json")

    ng.main(ng.Args(**{**vars(args), "removed_shard": 0}))
    ng.main(args)

    data = json.loads(args.out.read_text())
    assert [r["run"] for r in data["runs"]] == RUNS
    assert {r["run"]: r["best_model"] for r in data["runs"]} == WINNERS
    assert [len(r["pruned_during_run"]) for r in data["runs"]] == [84, 78, 73]
    pooled = data["pooled"]
    assert pooled["runs"] == RUNS
    assert pooled["n_trials"] == 3 * 7560
    assert pooled["winner_runs"] == {w: run for run, w in WINNERS.items()}
    assert set(pooled["elpd"]) == {*WINNERS.values(), *SEEDS}
    assert pooled["best_model"] == WINNERS["run1"]
    assert pooled["best_seed"] == SEEDS[0]
    assert pooled["vs_best_seed"][WINNERS["run2"]] == {"diff": 390.0, "se": 5.0}
    # Every run's other models, labelled by their run (names repeat across runs).
    assert len(pooled["others_at_end"]) == 3 + 6 + 6
    assert len(pooled["pruned_during_run"]) == 84 + 78 + 73
    assert all(label.split("__")[0] in RUNS for label in pooled["pruned_during_run"])
    assert set(pooled["pruned_order"]) == set(pooled["pruned_during_run"])
    assert set(pooled["others_order"]) == set(pooled["others_at_end"])
    # Figure 3: (example set, target) cells of all runs' responses.
    assert sum(c["n"] for c in data["conditions"]) == 3 * 7560
    # Figure 2 draws it: every winner and seed named, every run's models in one strip.
    import yaml
    from scripts.paper import figures

    names = yaml.safe_load(figures.Args().model_names.read_text())["number_game"]
    papers = yaml.safe_load(figures.Args().seed_papers.read_text())["number_game"]
    rows = figures.forest_rows("number_game", data, names, papers)
    assert [r["model"] for r in rows if r["kind"] == "winner"] == list(WINNERS.values())
    (strip,) = [r for r in rows if r["kind"] == "nonwinning"]
    assert sorted(set(strip["runs"])) == [0, 1, 2]
