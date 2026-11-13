"""Unit tests for ``scripts/paper/number_game_figure_data.py`` (no fitting):
the scoring against the best seed with a stand-in comparison, the condition
table, and what it reads from the committed dataset run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.paper import number_game_figure_data as ng

RUN1_DIR = ng.run_dir(ng.DEFAULT_RUNS_ROOT, "run1")
RUN3_DIR = ng.run_dir(ng.DEFAULT_RUNS_ROOT, "run3")


def test_the_runs_are_the_november_sweep_from_the_2016_papers_seeds():
    assert ng.RUN_ROOT == ng.here() / "data" / "results" / "number_game_dataset-nov26"


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
    assert live["winner"] == "span_fallback_rule_violation"
    assert len(live["others"]) == 2
    assert live["n_trials"] == 7560
    # Figure 2 draws them in admission order with the pruned models.
    assert len(ng.rm.pruned_order(final_dir, live["others"])) == 2

    seeds = json.loads((RUN1_DIR / "starting_models.json").read_text())["starting_models"]
    for seed in seeds:  # pruned in experiment 1: found in its pruned/ directory
        assert ng.sr.seed_file(RUN1_DIR, final_dir, seed).exists()


def test_the_removed_models_are_every_admitted_model_not_live_at_the_end_nor_a_seed():
    final_dir = ng.sr.final_experiment_dir(RUN1_DIR)
    live = ng.live_models(final_dir)
    seeds = json.loads((RUN1_DIR / "starting_models.json").read_text())["starting_models"]

    removed = ng.rm.removed_models(RUN1_DIR, final_dir,
                                   live=[live["winner"], *live["others"]], seeds=seeds)

    assert len(removed) == 87
    assert not set(removed) & {live["winner"], *live["others"], *seeds}
    assert all(r["outcome"] == "pruned" for r in removed.values())
    assert all(r["file"].exists() for r in removed.values())


def test_a_model_dropped_as_unfittable_is_found_in_models_not_pruned():
    final_dir = ng.sr.final_experiment_dir(RUN3_DIR)
    live = ng.live_models(final_dir)
    seeds = json.loads((RUN3_DIR / "starting_models.json").read_text())["starting_models"]

    removed = ng.rm.removed_models(RUN3_DIR, final_dir,
                                   live=[live["winner"], *live["others"]], seeds=seeds)

    assert removed["ordered_sweep_concepts"]["outcome"] == "dropped"
    assert removed["ordered_sweep_concepts"]["file"] == (
        RUN3_DIR / "experiment1" / "model_loop" / "models" / "ordered_sweep_concepts.py")
    # Run 3 reused the placeholder name iter2_candidate1 for two models.
    assert {"iter2_candidate1_experiment1", "iter2_candidate1_experiment3"} <= set(removed)


def test_the_models_pruned_at_the_end_of_experiment_3_keep_the_loops_elpd():
    final_dir = ng.sr.final_experiment_dir(RUN1_DIR)
    live = ng.live_models(final_dir)
    seeds = json.loads((RUN1_DIR / "starting_models.json").read_text())["starting_models"]
    removed = ng.rm.removed_models(RUN1_DIR, final_dir,
                                   live=[live["winner"], *live["others"]], seeds=seeds)

    recorded = ng.rm.recorded_scores(final_dir, removed)

    assert len(recorded) == 35
    assert recorded["round_number_kinship_exemplar"] == pytest.approx(-4255.512, abs=1e-4)
    # Exactly the models pruned (not retired) at the end of experiment 3.
    late = {n for n, r in removed.items() if r["outcome"] == "pruned"
            and r["context"] == "experiment3 end of experiment"
            and ng.rm.RETIRED_MARK not in r["detail"]}
    assert set(recorded) == late
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
WINNERS = {"run1": "span_fallback_rule_violation",
           "run2": "neighbourhood_doubt_three_example_memory",
           "run3": "exception_weighted_rule_rejection"}
SEEDS = ["bigelow_piantadosi_independent", "bigelow_piantadosi_mixture",
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
    assert [len(r["pruned_during_run"]) for r in data["runs"]] == [87, 82, 86]
    pooled = data["pooled"]
    assert pooled["runs"] == RUNS
    assert pooled["n_trials"] == 3 * 7560
    assert pooled["winner_runs"] == {w: run for run, w in WINNERS.items()}
    assert set(pooled["elpd"]) == {*WINNERS.values(), *SEEDS}
    assert pooled["best_model"] == WINNERS["run1"]
    assert pooled["best_seed"] == SEEDS[0]
    assert pooled["vs_best_seed"][WINNERS["run2"]] == {"diff": 390.0, "se": 5.0}
    # Every run's other models, labelled by their run (names repeat across runs).
    assert len(pooled["others_at_end"]) == 2 + 7 + 0
    assert len(pooled["pruned_during_run"]) == 87 + 82 + 86
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


POOLED_SETTINGS = {"method": "nuts_no_jitter", "tune": 4000, "draws": 1000, "chains": 4,
                   "target_accept": 0.95, "random_seed": 20261018}
RUN_SETTINGS = {"method": "warm_start_nuts", "tune": 200, "draws": 500, "chains": 4}


def _settings(kwargs, keys):
    return {k: kwargs.get(k) for k in keys}


def test_pooled_winners_and_seeds_take_the_pooled_settings_and_removed_models_the_runs(
        tmp_path, monkeypatch):
    """User decision 2026-11-01: at the runs' settings run 2's winner did not
    converge on the pooled data, so the pooled winners and seeds are fitted at
    the finals comparison's longer settings; each run's own comparison and
    every removed model, pooled or not, keep the runs' settings."""
    import src.model_comparison.posterior as posterior
    import src.models.pymc_inference as pymc_inference

    compared, predicted, removed = [], [], []

    def compare(responses, models_dir, *, names=None, **kwargs):
        compared.append((responses.parent.name, kwargs))
        top = max(names, key=_stand_in_elpd)
        return {n: {"elpd_loo": _stand_in_elpd(n), "dse_clustered": 0.0 if n == top else 5.0,
                    "loo_unreliable": False, "not_converged": False} for n in names}

    def score_removed_model(name, *args, **kwargs):
        removed.append((name, kwargs))
        return {"elpd": _stand_in_elpd(name)}

    def posterior_mean_p_yes(name, models_dir, responses, **kwargs):
        predicted.append((responses.parent.name, kwargs))
        return np.full(len(ng.sr._read_rows(responses)), 0.5)

    monkeypatch.setattr(posterior, "compare_table", compare)
    monkeypatch.setattr(pymc_inference, "fit_models_to_cache", lambda names, *a, **k: {})
    monkeypatch.setattr(pymc_inference, "clear_fit_cache", lambda: None)
    monkeypatch.setattr(ng, "score_removed_model", score_removed_model)
    monkeypatch.setattr(ng, "posterior_mean_p_yes", posterior_mean_p_yes)
    args = ng.Args(work_dir=tmp_path / "work", out=tmp_path / "number_game.json")

    ng.main(ng.Args(**{**vars(args), "removed_shard": 0}))
    ng.main(args)

    for where, kwargs in compared + predicted:
        expected = POOLED_SETTINGS if where == "pooled" else RUN_SETTINGS
        assert _settings(kwargs, expected) == expected, where
    assert {where for where, _ in compared} == {*RUNS, "pooled"}
    assert {where for where, _ in predicted} == {"pooled"}
    assert any("__" in name for name, _ in removed)  # the pooled extras
    assert all(_settings(k, RUN_SETTINGS) == RUN_SETTINGS for _, k in removed)
    data = json.loads(args.out.read_text())
    assert _settings(data["fit_settings"], RUN_SETTINGS) == RUN_SETTINGS
    assert _settings(data["pooled_fit_settings"], POOLED_SETTINGS) == POOLED_SETTINGS


def test_a_pooled_fit_task_fits_one_winner_or_seed_on_one_training_file(tmp_path, monkeypatch):
    """The pooled winners' and seeds' fits run as a Slurm array, one (model,
    training file) per task, into the pooled fit cache the assembly reads."""
    import src.models.pymc_inference as pymc_inference

    fitted = []

    def fit_models_cached(names, *, models_dir, responses_path, cache_dir, **kwargs):
        fitted.append((tuple(names), responses_path.name, cache_dir, kwargs))
        return {}

    monkeypatch.setattr(pymc_inference, "fit_models_cached", fit_models_cached)
    args = ng.Args(work_dir=tmp_path / "work", out=tmp_path / "number_game.json")
    n_tasks = (len(WINNERS) + len(SEEDS)) * 5  # all data and 4 CV folds

    for task in range(n_tasks):
        ng.main(ng.Args(**{**vars(args), "pooled_fit_task": task}))

    assert {(names, file) for names, file, _, _ in fitted} == {
        ((name,), file) for name in [*WINNERS.values(), *SEEDS]
        for file in ["responses.csv", *(f"train_fold{f}.csv" for f in range(4))]}
    assert len(fitted) == n_tasks
    assert {cache for _, _, cache, _ in fitted} == {args.work_dir / "pooled" / "fit_cache"}
    assert all(_settings(k, POOLED_SETTINGS) == POOLED_SETTINGS for *_, k in fitted)
    assert not args.out.exists()
    with pytest.raises(ValueError, match="pooled fit task"):
        ng.main(ng.Args(**{**vars(args), "pooled_fit_task": n_tasks}))
