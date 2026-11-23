"""The paper's Figures 2-4 (``scripts/paper/figures/``), drawn from the
figure data that ``sr_figure_data.py`` and ``rsa_figure_data.py`` write, the
conditions' reference-game displays (Figure 3) and the winners' backward eliminations (Figure 4)."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest
import yaml

from scripts.paper import figures
from scripts.paper.figures import fig2, fig3, fig4, style


def _domain_data(domain, winner="w", seed="s1"):
    winners = [winner, f"{winner}_b"]  # each run's own winner
    return {
        "domain": domain,
        "scoring": "ELPD on the final data",
        "runs": [
            {
                "run": run, "n_trials": 100, "best_model": winners[k], "best_seed": seed,
                "elpd": {winners[k]: -88.0 + k, seed: -100.0, "s2": -108.0},
                "vs_best_seed": {
                    winners[k]: {"diff": 12.0 + k, "se": 3.0},
                    seed: {"diff": 0.0, "se": 0.0},
                    "s2": {"diff": -8.0, "se": None},
                },
                "others_at_end": {"rival": 9.0 + k},
                "pruned_during_run": {"gone": -4.0 - k}, "pruned_order": {"gone": 0},
                "others_order": {"rival": 1},
            }
            for k, run in enumerate(["r1", "r2"])
        ],
        # All runs pooled: Figure 3, and Figure 2's view of subjective randomness.
        "pooled": {"best_model": winner, "best_seed": seed,
                   "winner_runs": {winners[0]: "r1", winners[1]: "r2"},
                   "elpd": {winners[0]: -80.0, winners[1]: -95.0, seed: -100.0, "s2": -130.0},
                   "vs_best_seed": {winners[0]: {"diff": 20.0, "se": 4.0}, winners[1]: {"diff": 5.0, "se": 2.0},
                                    seed: {"diff": 0.0, "se": 0.0}, "s2": {"diff": -30.0, "se": 6.0}},
                   "runs": ["r1", "r2"], "others_at_end": {"r1__x": 15.0}, "pruned_during_run": {"r2__y": -3.0},
                   "pruned_order": {"r2__y": 4}, "others_order": {"r1__x": 7}},
        "conditions": [
            {"n": 10, "observed": o, "best_model": o + 0.05, "best_seed": 0.5}
            for o in (0.2, 0.5, 0.8)
        ],
    }


def _setup(tmp_path):
    data_dir = tmp_path / "figure_data"
    data_dir.mkdir()
    for domain in ("subjective_randomness", "rsa"):
        data = _domain_data(domain)
        if domain == "rsa":  # Figure 2's second line: the best textbook model
            data["best_textbook"] = {"model": "s0", "diff": -10.0}
            data["promoted_seeds"] = ["s1"]  # drawn in the promoted seeds' colour
        (data_dir / f"{domain}.json").write_text(json.dumps(data))
    names = tmp_path / "model_names.yaml"
    names.write_text(yaml.safe_dump({d: NAMES for d in ("subjective_randomness", "rsa")}))
    papers = tmp_path / "seed_papers.yaml"
    papers.write_text(yaml.safe_dump({d: PAPERS for d in ("subjective_randomness", "rsa")}))
    # each run's history: its incumbent (best_model) at each step, and the models standing then
    for run, models in (("r1", ["x", "w"]), ("r2", ["gone", "rival", "y", "w_b"])):
        for exp in (1, 2, 3):
            path = tmp_path / "history" / run / f"experiment{exp}/model_loop/history.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps([{"best_model": models[-1], "standing": {m: 0 for m in models}}]))
    histories = {d: str(tmp_path / "history" / "{run}") for d in ("subjective_randomness", "rsa")}
    return figures.Args(figure_data=data_dir, out_dir=tmp_path / "out",
                        model_names=names, seed_papers=papers, pending_domains=["number_game"],
                        histories=histories)


NAMES = {"w": "Winner one", "w_b": "Winner two", "s1": "Seed one", "s2": "Seed two", "s0": "Textbook"}
PAPERS = {"s1": "Author & Other, 1999", "s2": None, "s0": "Frank & Goodman, 2012"}


def test_figure_2_is_drawn_with_the_number_game_pending(tmp_path):
    args = _setup(tmp_path)
    args.figures = ["2"]
    figures.main(args)
    assert not (args.out_dir / "fig2_elpd_horizontal.png").exists()  # one layout of Figure 2
    for ext in ("pdf", "png"):
        assert (args.out_dir / f"fig2_elpd.{ext}").stat().st_size > 0, ext


def test_figures_3_and_4_are_drawn_from_the_figure_data_and_eliminations(tmp_path):
    args = figures.Args(out_dir=tmp_path, figures=["3", "4"])
    figures.main(args)
    for name in ("fig3_effects", "fig4_ablation"):
        for ext in ("pdf", "png"):
            assert (args.out_dir / f"{name}.{ext}").stat().st_size > 0, (name, ext)


def test_only_the_figures_asked_for_are_drawn(tmp_path):
    args = _setup(tmp_path)
    args.figures = ["2"]
    figures.main(args)
    assert sorted(p.name for p in args.out_dir.iterdir()) == ["fig2_elpd.pdf", "fig2_elpd.png"]


def test_figure_2_is_saved_at_pnas_full_width(tmp_path):
    from PIL import Image
    args = _setup(tmp_path)
    args.figures = ["2"]
    figures.main(args)
    width_in = Image.open(args.out_dir / "fig2_elpd.png").size[0] / style.DPI
    assert width_in == pytest.approx(fig2.WIDTH_IN, abs=0.01) and fig2.WIDTH_IN == 7.0


def test_a_missing_domain_that_is_not_pending_raises(tmp_path):
    args = _setup(tmp_path)
    (args.figure_data / "rsa.json").unlink()
    with pytest.raises(FileNotFoundError, match="rsa"):
        figures.main(args)


def _fig3_tables():
    args = figures.Args()
    return fig3.load(args.figure_data, args.rsa_displays)


def _ng_conditions():
    import pandas as pd
    return pd.DataFrame({"number_set": ["2,4", "2,4"], "target": [6, 7], "n": [10, 12], "observed": [0.9, 0.1],
                         "best_model": [0.8, 0.2], "best_seed": [0.99, 0.01]})


def test_figure_3_swaps_in_a_named_models_predictions_for_the_same_conditions():
    preds = {"model": "m3", "conditions": [{"number_set": "2,4", "target": 7, "n": 12, "observed": 0.1, "p_yes": 0.3},
                                            {"number_set": "2,4", "target": 6, "n": 10, "observed": 0.9, "p_yes": 0.7}]}
    t = fig3.with_model(_ng_conditions(), preds, "m3")
    assert t.best_model.to_list() == [0.7, 0.3] and t.best_seed.to_list() == [0.99, 0.01]


def test_a_named_models_predictions_for_other_conditions_or_another_model_raise():
    preds = {"model": "m3", "conditions": [{"number_set": "2,4", "target": 6, "n": 10, "observed": 0.9, "p_yes": 0.7},
                                            {"number_set": "2,4", "target": 8, "n": 12, "observed": 0.1, "p_yes": 0.3}]}
    with pytest.raises(RuntimeError, match="conditions"):
        fig3.with_model(_ng_conditions(), preds, "m3")
    with pytest.raises(RuntimeError, match="m2"):
        fig3.with_model(_ng_conditions(), {**preds, "model": "m3"}, "m2")


def test_figure_3s_best_winner_fits_each_domains_conditions_better_than_its_best_seed():
    for d, cond in _fig3_tables().items():
        assert fig3.r2(cond.observed, cond.best_model, cond.n) > fig3.r2(cond.observed, cond.best_seed, cond.n), d


def test_figure_3s_reference_games_are_the_preregistered_experiments_only():
    assert set(_fig3_tables()["rsa"].dataset) == {"prereg"}


def test_in_figure_3s_reference_game_people_pick_the_object_with_only_the_words_feature():
    d = fig3.reference_game(_fig3_tables()["rsa"])
    only_word = [k for k, cls in zip(d.index, d.choice_classes) if fig3.GAME[cls.index(k)] == [1, 0]]
    assert d.observed.idxmax() == only_word[0]


def test_figure_3s_alternation_curve_peaks_above_a_fair_coins_rate():
    t = fig3.alternation_curve(_fig3_tables()["subjective_randomness"])
    assert t.observed.idxmax() > (fig3.SR_LENGTH - 1) / 2


def _preds(observed):
    return {"observed": observed, "best_seed": observed, "best_model": observed}


def test_figure_3s_streak_choices_are_of_the_one_constant_sequence_by_its_rivals_switch_rate():
    import pandas as pd
    sr = pd.DataFrame({
        "pair": [["HHHHH", "HTHTT"], ["THTHH", "TTTTT"],  # kept: the constant sequence first, then second
                 ["HHHH", "HTHT"], ["HHHHH", "TTTTT"], ["HTHTH", "HHTTH"]],  # too short; both; neither
        "n": [10, 30, 5, 5, 5], **_preds([0.2, 0.9, 0.5, 0.5, 0.5])})
    t = fig3.streak_choices(sr)
    assert len(t) == 1 and t.n.iloc[0] == 40
    assert t.switch_rate.iloc[0] == pytest.approx(0.75)  # 3 switches in 4
    assert t.observed.iloc[0] == pytest.approx((10 * 0.2 + 30 * 0.1) / 40)  # the second's chosen 1 - 0.9


def test_in_figure_3_people_choose_a_constant_sequence_at_a_flat_rate_the_best_seed_does_not():
    t = fig3.streak_choices(_heldout_tables()["subjective_randomness"])
    assert np.ptp(t.observed) < 0.06 and np.ptp(t.best_model) < 0.06 and np.ptp(t.best_seed) > 0.15


def _rsa_display(display, objects, utterance, classes, n, observed):
    return [{"display": display, "cls": k, "n": n, "objects": objects, "utterance": utterance,
             "choice_classes": classes, **_preds(o)} for k, o in enumerate(observed)]


def test_figure_3s_fits_three_roles_are_per_object_trial_weighted_and_clustered_by_display():
    import pandas as pd
    rsa = pd.DataFrame(
        _rsa_display(0, [[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0]], 0.0, [0, 1, 2, 3], 10, [0.5, 0.3, 0.1, 0.1])
        + _rsa_display(1, [[1, 0], [1, 0], [1, 1], [0, 1]], 0.0, [0, 0, 1, 2], 30, [0.6, 0.3, 0.1])  # two copies
        + _rsa_display(2, [[1, 1], [1, 1], [1, 1]], 0.0, [0, 0, 0], 5, [1.0])  # all fitting alike: not shown
        + _rsa_display(3, [[1, 0], [1, 1], [0, 1]], 0.0, [0, 1, 2], 7, [0.8, 0.1, 0.1])  # the word fits two
        + _rsa_display(4, [[1, 0], [1, 1], [0, 1]], float("nan"), [0, 1, 2], 7, [0.2, 0.5, 0.3]))  # no word
    t = fig3.fits_three_roles(rsa)
    assert list(t.index) == ["doesn't fit the word", "fewest features", "in between", "most features"]
    assert t.trials.to_dict() == {"doesn't fit the word": 40, "fewest features": 40, "in between": 10,
                                  "most features": 40}
    # fewest features: display 0's object (0.5, 10 trials) and display 1's two copies (0.6 / 2 each, 30 trials)
    m = (10 * 0.5 + 2 * 30 * 0.3) / 70
    assert t.observed["fewest features"] == pytest.approx(m)
    assert t.observed["in between"] == pytest.approx(0.3)
    se = np.sqrt((10 * (0.5 - m)) ** 2 + (2 * 30 * (0.3 - m)) ** 2) / 70  # each display one cluster
    assert t.ci["fewest features"] == pytest.approx(1.96 * se)


def test_in_figure_3_when_the_word_fits_three_objects_the_seed_overrates_the_in_between_one():
    import pandas as pd
    from scripts.paper import fig3_patterns
    rsa = _fig3_tables()["rsa"]
    t = fig3.fits_three_roles(rsa)
    feats = pd.DataFrame([fig3_patterns._rsa(c) for c in rsa.itertuples()], index=rsa.index)
    group = rsa[feats["objects the word fits"] == "3"]
    alike = rsa[(feats["objects the word fits"] == "3") & (feats["the object"] == "all fitting alike")]
    assert t.trials.max() == group.drop_duplicates("display").n.sum() - alike.drop_duplicates("display").n.sum()
    assert t.observed.idxmax() == "fewest features"
    near = (t.best_model - t.observed).abs() <= t.ci
    assert near.drop("doesn't fit the word").all()
    assert t.best_seed["in between"] - t.observed["in between"] > 2 * t.ci["in between"]


def test_figure_3s_row_c_patterns_are_candidates_of_the_search(tmp_path):
    fig3.check_row_c(figures.Args().figure_data / "fig3_patterns.json")
    found = {d: [{"feature": "f", "value": "v", "candidate": True}] for d in fig3.ROW_C}
    (tmp_path / "p.json").write_text(json.dumps(found))
    with pytest.raises(RuntimeError, match="not one of the search's candidates"):
        fig3.check_row_c(tmp_path / "p.json")


def test_figure_3_flags_no_panel():
    assert fig3.TO_FIX == {}


def _heldout_tables():
    return fig3.load_heldout(figures.Args().figure_data)


def test_figure_3s_subjective_randomness_rows_a_and_b_are_figure_2s_leave_one_out():
    pooled = json.loads((figures.Args().figure_data / "subjective_randomness.json").read_text())["pooled"]
    t = _heldout_tables()["subjective_randomness"]
    assert t.n.sum() == pooled["n_trials"]
    for role in fig3.MODELS:  # each pair's summed loo_i: together, Figure 2's ELPD-LOO
        assert t[f"{role}_lpd"].sum() == pytest.approx(pooled["elpd"][pooled[role]], abs=1e-6), role


def test_figure_3s_number_game_rows_a_and_b_are_figure_2s_unseen_items():
    test = json.loads(figures.Args().ng_unseen.read_text())["unseen_items"]
    t = _heldout_tables()["number_game"]
    assert len(t) == test["n_items"] and t.n.sum() == test["n_responses"]
    names = {"best_model": fig3.COLUMN_MODEL["number_game"][0], "best_seed": test["best_seed"]}
    for role, name in names.items():
        assert t[f"{role}_lpd"].sum() == pytest.approx(test["models"][name]["lpd"], rel=1e-9), role


def test_figure_3s_held_out_predictions_must_be_of_the_columns_models(tmp_path):
    args = figures.Args()
    for path in args.figure_data.glob("*.json"):
        (tmp_path / path.name).write_text(path.read_text())
    p = tmp_path / "subjective_randomness_heldout_conditions.json"
    held = json.loads(p.read_text())
    held["models"]["best_seed"] = "another_seed"
    p.write_text(json.dumps(held))
    with pytest.raises(RuntimeError, match="another_seed"):
        fig3.load_heldout(tmp_path)


def test_figure_3s_classic_number_game_profiles_cover_every_number_on_all_items():
    full = fig3.all_items(_fig3_tables()["number_game"], _heldout_tables()["number_game"])
    for ex, p in fig3.ng_profiles(full).items():
        examples = {int(v) for v in ex.split(",")}
        assert p.drop(index=[t for t in examples if t in p.index and p.observed.isna()[t]]).notna().all().all(), ex
        assert p.observed[[10, 20, 30, 40, 50]].notna().all(), ex


def test_all_items_joins_the_seen_and_unseen_items_and_raises_on_an_item_in_both():
    import pandas as pd
    seen = _ng_conditions()
    unseen = pd.DataFrame({"number_set": ["2,4"], "target": [8], "n": [9], **_preds([0.5]), "best_model_lpd": [-1.0]})
    t = fig3.all_items(seen, unseen)
    assert sorted(t.target) == [6, 7, 8] and set(t.columns) == set(seen.columns)
    with pytest.raises(RuntimeError, match="both"):
        fig3.all_items(seen, unseen.assign(target=7))


def test_figure_3s_generalization_profiles_leave_a_number_without_data_empty():
    import pandas as pd
    targets = [t for t in range(1, 101) if t != 50]
    ng = pd.DataFrame({"number_set": "100", "target": targets, "n": 10, **_preds([0.3] * len(targets))})
    p = fig3.ng_profiles(ng, ("100",))["100"]
    assert list(p.index) == list(range(1, 101))
    assert np.isnan(p.observed[50]) and np.isnan(p.best_model[50]) and p.observed.notna().sum() == 99


def test_figure_3s_one_example_roles_are_trial_weighted_and_clustered_by_example_set():
    import pandas as pd
    ng = pd.DataFrame({
        "number_set": ["6", "6", "6", "6", "6", "9", "9", "2,4"],
        "target": [6, 12, 3, 5, 7, 3, 10, 3],  # the example; a multiple; a factor; other; other; a factor; other
        "n": [10, 10, 10, 10, 30, 20, 10, 10], **_preds([1.0, 0.6, 0.5, 0.1, 0.3, 0.2, 0.0, 0.9])})
    t = fig3.one_example_roles(ng)
    assert list(t.index) == ["the example", "a multiple", "a factor", "other"]
    assert t.trials.to_dict() == {"the example": 10, "a multiple": 10, "a factor": 30, "other": 50}
    m = (10 * 0.5 + 20 * 0.2) / 30
    assert t.observed["a factor"] == pytest.approx(m)
    se = np.sqrt((10 * (0.5 - m)) ** 2 + (20 * (0.2 - m)) ** 2) / 30  # each example set one cluster
    assert t.ci["a factor"] == pytest.approx(1.96 * se)


def test_in_figure_3_after_one_example_the_seed_underrates_its_factors_on_held_out_items():
    t = fig3.one_example_roles(_heldout_tables()["number_game"]).loc["a factor"]
    assert t.observed - t.best_seed > 2 * t.ci
    assert abs(t.best_model - t.observed) < abs(t.best_seed - t.observed)


def test_figure_3s_row_c_patterns():
    assert fig3.ROW_C == {"subjective_randomness": ("a pure streak in the pair", "one"),
                          "rsa": ("objects the word fits", "3"),
                          "number_game": ("after one example, the number is", "a factor")}


def test_every_mechanism_figure_4_drops_or_keeps_has_a_name():
    args = figures.Args()
    names = yaml.safe_load(args.mechanisms.read_text())
    for domain, path in fig4.sources(args.figure_data).items():
        rounds = json.loads(path.read_text())["rounds"]
        assert set(rounds[0]["mechanisms"]) <= set(names[domain]), domain


def test_a_mock_domain_is_marked_mock_in_its_panel_title():
    assert style.panel_title("rsa", mock_domains=["rsa"]) == "Reference games (MOCK DATA)"
    assert style.panel_title("rsa", mock_domains=["number_game"]) == "Reference games"


def test_mock_domains_must_be_drawn_domains(tmp_path):
    args = _setup(tmp_path)
    args.mock_domains = ["subjective_randomnes"]
    with pytest.raises(ValueError, match="subjective_randomnes"):
        figures.main(args)


def _labelled(rows):
    return [(r["kind"], r["label"], r["diff"], r["se"]) for r in rows if "diff" in r]


def test_runs_on_one_test_collapse_into_one_forest():
    rows = figures.forest_rows("rsa", _domain_data("rsa"), NAMES, PAPERS)
    # Winners first (best first), then the strips, then the seeds.
    assert _labelled(rows) == [
        ("winner", "Winner two", 13.0, 3.0),
        ("winner", "Winner one", 12.0, 3.0),
        ("seed", "Seed one", 0.0, 0.0),
        ("seed", "Seed two", -8.0, None),  # no paired SE recorded: drawn hollow
    ]
    strips = {r["kind"]: r["values"] for r in rows if "values" in r}
    assert strips == {"nonwinning": [-4.0, 9.0, -5.0, 10.0]}  # run by run, in admission order


def test_the_non_winning_models_of_every_run_share_one_strip_in_admission_order():
    data = _domain_data("rsa")
    for k, run in enumerate(data["runs"]):
        run["pruned_during_run"] = {f"late_{k}": 4.0 - k, f"early_{k}": -k}
        run["pruned_order"] = {f"late_{k}": 9, f"early_{k}": 2}
        run["others_order"] = {"rival": 5}  # admitted between them, still live
    rows = figures.forest_rows("rsa", data, NAMES, PAPERS)
    (strip,) = [r for r in rows if "values" in r]
    assert strip["kind"] == "nonwinning"
    # Run by run, in admission order: early, rival (live), late.
    assert strip["values"] == [0.0, 9.0, 4.0, -1.0, 10.0, 3.0]
    assert strip["models"] == ["early_0", "rival", "late_0", "early_1", "rival", "late_1"]
    assert strip["runs"] == [0, 0, 0, 1, 1, 1]
    assert [(r["model"], r["run"]) for r in rows if r["kind"] == "winner"] == [("w_b", 1), ("w", 0)]
    # Each run's models from 0 (first admitted) to 1 (last).
    assert strip["progress"] == [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]


def test_pooled_domains_are_drawn_from_all_runs_pooled():
    # Each run scored on its own data, so its runs are not on one test.
    rows = figures.forest_rows("subjective_randomness", _domain_data("subjective_randomness"), NAMES, PAPERS)
    assert _labelled(rows) == [("winner", "Winner one", 20.0, 4.0), ("winner", "Winner two", 5.0, 2.0),
                               ("seed", "Seed one", 0.0, 0.0), ("seed", "Seed two", -30.0, 6.0)]
    (strip,) = [r for r in rows if "values" in r]
    assert strip["values"] == [15.0, -3.0]  # the pooled labels name each model's run


def test_a_labelled_model_without_a_name_raises():
    with pytest.raises(KeyError, match="'w_b' is not in model_names.yaml"):
        figures.forest_rows("rsa", _domain_data("rsa"), {"w": "Winner one", "s1": "x", "s2": "y"},
                            PAPERS)


def test_a_starting_model_not_in_seed_papers_raises():
    with pytest.raises(KeyError, match="'s2' is not in seed_papers.yaml"):
        figures.forest_rows("rsa", _domain_data("rsa"), NAMES, {"s1": "Author, 1999"})


def test_starting_models_are_named_by_their_paper_one_name_for_a_papers_models():
    def seed(model, paper):
        return {"kind": "seed", "model": model, "label": model.upper(), "diff": 0.0, "paper": paper}

    columns = [seed("a", "Frank & Goodman, 2012"), seed("c", "Hahn & Warren, 2009"),
               seed("b", "Frank & Goodman, 2012"), seed("d", None)]
    assert fig2.seed_names(columns) == [("Frank & Goodman (2012)", ["a", "b"]), ("Hahn & Warren (2009)", ["c"]),
                                        ("D", ["d"])]


def test_reference_games_are_drawn_against_the_best_textbook_model():
    data = {**_domain_data("rsa"), "best_textbook": {"model": "s0", "diff": -10.0}, "promoted_seeds": ["s1"]}
    rows = fig2.rsa_vs_textbook(figures.forest_rows("rsa", data, NAMES, PAPERS), data, NAMES, PAPERS)
    # Every value shifted by the textbook model's +10; the promoted seed (s1) leaves the seeds' columns.
    assert [(r["kind"], r["label"], r["diff"]) for r in rows if r["kind"] in ("winner", "seed")] == [
        ("winner", "Winner two", 23.0), ("winner", "Winner one", 22.0),
        ("seed", "Seed two", 2.0), ("seed", "Textbook", 0.0)]
    (promoted,) = [r for r in rows if r["kind"] == "promoted"]
    assert promoted["values"] == [10.0]  # the best promoted seed: the pre-registered bar
    assert [r["se"] for r in rows if r["kind"] == "seed"] == [None, 0.0]  # the seeds' paired SEs dropped


def test_each_runs_trajectory_is_its_incumbent_at_its_test_value_as_the_run_admits_models():
    rows = figures.forest_rows("rsa", _domain_data("rsa"), NAMES, PAPERS)  # strip: gone, rival per run
    steps = [{"best_model": "s1", "standing": {"s1": 0}},  # the seed, before any model is admitted
             {"best_model": "rival", "standing": {"s1": 0, "gone": 0, "rival": 0}},
             {"best_model": "w", "elpd_loo": {"rival": 0, "w": 0}}]
    paths = fig2.incumbent_paths("rsa", rows, [steps, steps[:1]])
    # (share of the run's strip admitted, the incumbent's value): s1 at 0, rival (+9) once both are in.
    assert paths[0] == [(0.0, 0.0), (1.0, 9.0), (1.0, 12.0)]
    assert paths[1] == [(0.0, 0.0)]


def test_without_avenir_figure_2_uses_a_close_sans_serif(monkeypatch, capsys):
    monkeypatch.setattr(style, "AVENIR_PATHS", [])
    import matplotlib.pyplot as plt
    with plt.rc_context():
        style.setup_style()
        family = plt.rcParams["font.family"]
    assert family[0] != "Avenir" and family[-1] == "DejaVu Sans"
    assert "Avenir not found" in capsys.readouterr().out


def _unseen():
    def test_set(best_seed, shift):
        record = {"best_seed": best_seed, "n_responses": 1000, "models": {
            m: {"lpd": lpd + shift, "vs_best_seed": {"diff": diff, "se": se}}
            for m, (lpd, diff, se) in {"w": (-50.0, 30.0, 5.0), "w_b": (-60.0, 20.0, 4.0),
                                       "s1": (-90.0, -10.0, 3.0), "s2": (-80.0, 0.0, 0.0)}.items()}}
        record["strip"] = {"r1__x": 12.0, "r2__y": -7.0}
        return record
    return {"winner_runs": {"w": "r1", "w_b": "r2"}, "unseen_items": test_set("s2", 0.0),
            "unseen_example_sets": test_set("s2", 40.0)}


def test_figure_2s_number_game_is_its_models_scored_on_unseen_items():
    data = _domain_data("number_game")
    view = figures.number_game_unseen_view(data, _unseen(), "unseen_items")
    rows = figures.forest_rows("number_game", view, NAMES, PAPERS)
    assert _labelled(rows) == [("winner", "Winner one", 30.0, 5.0), ("winner", "Winner two", 20.0, 4.0),
                               ("seed", "Seed two", 0.0, 0.0), ("seed", "Seed one", -10.0, 3.0)]
    (strip,) = [r for r in rows if "values" in r]
    assert strip["values"] == [12.0, -7.0]
    sets = figures.number_game_unseen_view(data, _unseen(), "unseen_example_sets")
    assert sets["pooled"]["elpd"]["s2"] == -40.0  # each test set against its own best seed
    # Figures 3 and 4 keep the cross-validated comparison.
    assert data["pooled"]["best_seed"] == "s1" and data["pooled"]["others_at_end"] == {"r1__x": 15.0}


def test_unseen_scores_of_other_winners_raise():
    unseen = _unseen()
    unseen["winner_runs"] = {"w": "r1", "other": "r2"}
    with pytest.raises(RuntimeError, match="winners"):
        figures.number_game_unseen_view(_domain_data("number_game"), unseen, "unseen_items")


def test_an_unseen_strip_missing_a_model_raises():
    unseen = _unseen()
    del unseen["unseen_items"]["strip"]["r2__y"]
    with pytest.raises(KeyError, match="r2__y"):
        figures.number_game_unseen_view(_domain_data("number_game"), unseen, "unseen_items")


def test_the_number_game_is_titled_number_game():
    assert style.panel_title("number_game", mock_domains=[]) == "Number game"


def _elimination():
    """Four mechanisms, two dropped: a (cost 1, se 2 vs the full model), then b (cost 3, se 1.5 vs the model
    without a), then the stop. Each round's se_vs_start differs from its step's se, so the test tells them apart."""
    def cand(m, elpd, cur, se, se_vs_start):
        return {"mechanism": m, "elpd_cv": elpd, "cost": cur - elpd, "se": se, "se_vs_start": se_vs_start}
    return {"rounds": [
        {"mechanisms": ["a", "b", "c", "d"], "elpd_cv": -100.0, "se_vs_start": 0.0,
         "candidates": [cand("a", -101.0, -100.0, 2.0, 2.0), cand("b", -110.0, -100.0, 5.0, 5.0)]},
        {"mechanisms": ["b", "c", "d"], "elpd_cv": -101.0, "se_vs_start": 2.0,
         "candidates": [cand("b", -104.0, -101.0, 1.5, 6.0), cand("c", -140.0, -101.0, 8.0, 9.0)]},
        {"mechanisms": ["c", "d"], "elpd_cv": -104.0, "se_vs_start": 6.0,
         "candidates": [cand("c", -130.0, -104.0, 9.0, 12.0), cand("d", -120.0, -104.0, 7.0, 10.0)]},
    ]}


def test_figure_4s_path_bars_are_the_stopping_rules_se_vs_the_previous_round():
    path, _ = fig4.ablation_points(_elimination())
    assert [p["n"] for p in path] == [4, 3, 2]
    assert [p["y"] for p in path] == [0.0, -1.0, -4.0]  # ELPD-CV vs. the full model
    assert [p["dropped"] for p in path] == [None, "a", "b"]
    assert [p["se"] for p in path] == [None, 2.0, 1.5]  # each drop's own se, not se_vs_start (2.0, 6.0)


def test_figure_4s_kept_column_is_each_removal_vs_the_final_model():
    _, kept = fig4.ablation_points(_elimination())
    assert [k["mechanism"] for k in kept] == ["d", "c"]  # cheapest removal first
    assert [k["y"] for k in kept] == [-16.0, -26.0]  # zero: the final model
    assert [k["se"] for k in kept] == [7.0, 9.0]  # the stopping rule's se vs the final model


def test_figure_4_shows_each_domains_stopping_rule():
    """Every drawn step cost under 2 of its SEs; the kept column's cheapest removal costs at least 2."""
    for domain, path in fig4.sources(figures.Args().figure_data).items():
        steps, kept = fig4.ablation_points(json.loads(path.read_text()))
        for prev, step in itertools.pairwise(steps):
            assert prev["y"] - step["y"] < 2 * step["se"], (domain, step["dropped"])
        assert -kept[0]["y"] >= 2 * kept[0]["se"], domain


# Fig. 4 prints as a figure* at width=0.9\linewidth; pnas-new.cls sets \textwidth to 512 TeX pt (1/72.27 in)
FIG4_PRINT_PT = 0.9 * 512 / 72.27 * 72  # its printed width in PDF points (459.2)


def _pdf_font_sizes(path):
    """Each text span's (font size in PDF points, text): its Tf size times the scale of its text and current
    transformation matrices."""
    from pypdf import PdfReader
    page = PdfReader(path).pages[0]
    spans = []

    def visit(text, cm, tm, font, size):
        if text.strip():
            a, b = cm[0] * tm[0] + cm[2] * tm[1], cm[1] * tm[0] + cm[3] * tm[1]
            c, d = cm[0] * tm[2] + cm[2] * tm[3], cm[1] * tm[2] + cm[3] * tm[3]
            spans.append((size * abs(a * d - b * c) ** 0.5, text))
    page.extract_text(visitor_text=visit)
    return float(page.mediabox.width), spans


def test_all_of_figure_4s_text_prints_at_6_to_12_pt(tmp_path):
    """PNAS: no text under 6 pt (or over 12) at the printed size, the number cards' digits included."""
    figures.main(figures.Args(out_dir=tmp_path, figures=["4"]))
    width, spans = _pdf_font_sizes(tmp_path / "fig4_ablation.pdf")
    printed = [(size * FIG4_PRINT_PT / width, text) for size, text in spans]
    assert {"16", "8", "2", "64"} <= {text.strip() for _, text in printed}  # the icon's digits are measured
    assert [(pt, text) for pt, text in printed if not 6 <= pt <= 12] == []


def test_figure_4s_title_icons_clear_their_axes_labels(tmp_path, monkeypatch):
    """Each title's icon (the axes beyond the 3 x 2 grid) overlaps no y label or tick label of the path axes."""
    drawn = []
    monkeypatch.setattr(fig4.plt, "close", drawn.append)
    figures.main(figures.Args(out_dir=tmp_path, figures=["4"]))
    (fig,) = drawn
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    paths, icons = fig.axes[0:6:2], fig.axes[6:]
    def drawn_ticks(axis, lims):  # matplotlib keeps (undrawn) ticks beyond the view limits
        lo, hi = sorted(lims)
        return [t.label1 for t in axis.get_major_ticks() if lo <= t.get_loc() <= hi]
    labels = [t.get_window_extent(renderer) for ax in paths
              for t in [ax.yaxis.label, *drawn_ticks(ax.yaxis, ax.get_ylim()), *drawn_ticks(ax.xaxis, ax.get_xlim())]
              if t.get_text()]
    for icon in icons:
        box = icon.get_window_extent(renderer)
        assert not [b for b in labels if b.overlaps(box)], icon
