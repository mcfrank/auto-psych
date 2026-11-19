"""The paper's Figures 2-4 (``scripts/paper/figures/``), drawn from the
figure data that ``sr_figure_data.py`` and ``rsa_figure_data.py`` write and
the hand-written pseudocode specs."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import yaml

from scripts.paper import figures
from scripts.paper.figures import fig2, legacy, style


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


def _setup(tmp_path, *, winner_in_spec="w", corrupt_sha=False):
    data_dir = tmp_path / "figure_data"
    data_dir.mkdir()
    spec_dir = tmp_path / "pseudocode"
    spec_dir.mkdir()
    for domain in ("subjective_randomness", "rsa"):
        data = _domain_data(domain)
        if domain == "rsa":  # Figure 2's second line: the best textbook model
            data["best_textbook"] = {"model": "s0", "diff": -10.0}
            data["promoted_seeds"] = ["s1"]  # drawn in the promoted seeds' colour
        (data_dir / f"{domain}.json").write_text(json.dumps(data))
        source = tmp_path / f"{domain}_model.py"
        source.write_text("model = None\n")
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        (spec_dir / f"{domain}.yaml").write_text(yaml.safe_dump({
            "domain": domain, "model": winner_in_spec, "source": str(source),
            "sha256": "0" * 64 if corrupt_sha else sha, "title": f"{domain} title",
            "lines": [{"code": "for each x:", "note": "a note"}, {"code": "  y = f(x)"}],
        }, allow_unicode=True))
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
    return figures.Args(figure_data=data_dir, pseudocode=spec_dir, out_dir=tmp_path / "out",
                        model_names=names, seed_papers=papers, pending_domains=["number_game"],
                        histories=histories)


NAMES = {"w": "Winner one", "w_b": "Winner two", "s1": "Seed one", "s2": "Seed two", "s0": "Textbook"}
PAPERS = {"s1": "Author & Other, 1999", "s2": None, "s0": "Frank & Goodman, 2012"}


def test_all_three_figures_are_drawn_with_the_number_game_pending(tmp_path):
    args = _setup(tmp_path)
    figures.main(args)
    assert not (args.out_dir / "fig2_elpd_horizontal.png").exists()  # one layout of Figure 2
    for name in ("fig2_elpd", "fig3_conditions", "fig4_pseudocode"):
        for ext in ("pdf", "png"):
            assert (args.out_dir / f"{name}.{ext}").stat().st_size > 0, (name, ext)


def test_only_the_figures_asked_for_are_drawn(tmp_path):
    args = _setup(tmp_path)
    args.figures = ["2"]
    (args.pseudocode / "rsa.yaml").unlink()  # Figure 4's specs are not read
    figures.main(args)
    assert sorted(p.name for p in args.out_dir.iterdir()) == ["fig2_elpd.pdf", "fig2_elpd.png"]


def test_figure_2_is_saved_at_pnas_full_width(tmp_path):
    from PIL import Image
    args = _setup(tmp_path)
    args.figures = ["2"]
    figures.main(args)
    width_in = Image.open(args.out_dir / "fig2_elpd.png").size[0] / 300
    assert width_in == pytest.approx(fig2.WIDTH_IN, abs=0.01) and fig2.WIDTH_IN == 7.0


def test_a_missing_domain_that_is_not_pending_raises(tmp_path):
    args = _setup(tmp_path)
    (args.figure_data / "rsa.json").unlink()
    with pytest.raises(FileNotFoundError, match="rsa"):
        figures.main(args)


def test_pseudocode_whose_source_changed_raises(tmp_path):
    with pytest.raises(RuntimeError, match="sha256"):
        figures.main(_setup(tmp_path, corrupt_sha=True))


def test_pseudocode_of_another_model_than_the_pooled_best_model_raises(tmp_path):
    with pytest.raises(RuntimeError, match="best model on the pooled data"):
        figures.main(_setup(tmp_path, winner_in_spec="other"))


def test_fit_statistics_weight_each_condition_by_its_trials():
    observed = np.array([0.0, 1.0, 0.5])
    predicted = np.array([0.0, 0.5, 0.5])
    n = np.array([1, 3, 1])
    r, rmse = legacy.weighted_fit_stats(observed, predicted, n)
    assert rmse == pytest.approx(np.sqrt(3 * 0.25 / 5))
    assert -1.0 <= r <= 1.0
    r_unit, _ = legacy.weighted_fit_stats(observed, observed, n)
    assert r_unit == pytest.approx(1.0)
    r_flat, _ = legacy.weighted_fit_stats(observed, np.full(3, 0.5), n)
    assert np.isnan(r_flat)


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
