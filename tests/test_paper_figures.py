"""The paper's Figures 2-4 (``scripts/paper/figures.py``), drawn from the
figure data that ``sr_figure_data.py`` and ``rsa_figure_data.py`` write and
the hand-written pseudocode specs."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import yaml

from scripts.paper import figures


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
            }
            for k, run in enumerate(["r1", "r2"])
        ],
        # All runs pooled: Figure 3, and Figure 2's view of subjective randomness.
        "pooled": {"best_model": winner, "best_seed": seed,
                   "winner_runs": {winners[0]: "r1", winners[1]: "r2"},
                   "elpd": {winners[0]: -80.0, winners[1]: -95.0, seed: -100.0, "s2": -130.0}},
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
    return figures.Args(figure_data=data_dir, pseudocode=spec_dir, out_dir=tmp_path / "out",
                        model_names=names, seed_papers=papers, pending_domains=["number_game"])


NAMES = {"w": "Winner one", "w_b": "Winner two", "s1": "Seed one", "s2": "Seed two"}
PAPERS = {"s1": "Author & Other, 1999", "s2": None}


def test_all_three_figures_are_drawn_with_the_number_game_pending(tmp_path):
    args = _setup(tmp_path)
    figures.main(args)
    for name in ("fig2_elpd", "fig2_elpd_horizontal", "fig3_conditions", "fig4_pseudocode"):
        for ext in ("pdf", "png"):
            assert (args.out_dir / f"{name}.{ext}").stat().st_size > 0, (name, ext)


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
    r, rmse = figures.weighted_fit_stats(observed, predicted, n)
    assert rmse == pytest.approx(np.sqrt(3 * 0.25 / 5))
    assert -1.0 <= r <= 1.0
    r_unit, _ = figures.weighted_fit_stats(observed, observed, n)
    assert r_unit == pytest.approx(1.0)
    r_flat, _ = figures.weighted_fit_stats(observed, np.full(3, 0.5), n)
    assert np.isnan(r_flat)


def test_rsa_chains_are_labelled_as_runs_like_every_other_domain():
    assert figures.run_label("rsa", "c0") == "Run 1"
    assert figures.run_label("rsa", "c2") == "Run 3"
    assert figures.run_label("subjective_randomness", "run2") == "Run 2"


def test_a_mock_domain_is_marked_mock_in_its_panel_title():
    assert figures.panel_title("rsa", mock_domains=["rsa"]) == "Reference games (MOCK DATA)"
    assert figures.panel_title("rsa", mock_domains=["number_game"]) == "Reference games"


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
    assert strips == {"nonwinning": None}  # pruned not scored: pending


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
    # Each run's models from 0 (first admitted) to 1 (last).
    assert strip["progress"] == [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
    # Each model's run, for its colour.
    assert strip["runs"] == [0, 0, 0, 1, 1, 1]
    assert [(r["model"], r["run"]) for r in rows if r["kind"] == "winner"] == [("w_b", 1), ("w", 0)]


def test_subjective_randomness_is_drawn_from_all_runs_pooled():
    data = _domain_data("subjective_randomness")
    rows = figures.forest_rows("subjective_randomness", data, NAMES, PAPERS)
    # Before the pooled paired SEs exist: differences from the pooled ELPDs.
    assert _labelled(rows) == [("winner", "Winner one", 20.0, None),
                               ("winner", "Winner two", 5.0, None),
                               ("seed", "Seed one", 0.0, 0.0), ("seed", "Seed two", -30.0, None)]
    data["pooled"]["vs_best_seed"] = {"w": {"diff": 20.0, "se": 4.0}, "w_b": {"diff": 5.0, "se": 2.0},
                                      "s1": {"diff": 0.0, "se": 0.0}, "s2": {"diff": -30.0, "se": 6.0}}
    data["pooled"].update(runs=["r1", "r2"], others_at_end={"r1__x": 15.0},
                          pruned_during_run={"r2__y": -3.0}, pruned_order={"r2__y": 4},
                          others_order={"r1__x": 7})
    rows = figures.forest_rows("subjective_randomness", data, NAMES, PAPERS)
    assert [r["se"] for r in rows if "se" in r] == [4.0, 2.0, 0.0, 6.0]
    (strip,) = [r for r in rows if "values" in r]
    assert strip["values"] == [15.0, -3.0]
    # The pooled labels name each model's run.
    assert strip["runs"] == [0, 1]


def test_the_number_game_is_drawn_from_all_runs_pooled():
    # Each run drew its own items, so its runs are not on one test.
    data = _domain_data("number_game")
    rows = figures.forest_rows("number_game", data, NAMES, PAPERS)
    assert _labelled(rows)[0] == ("winner", "Winner one", 20.0, None)
    assert figures.best_seed_elpd("number_game", data) == -100.0


def test_a_labelled_model_without_a_name_raises():
    with pytest.raises(KeyError, match="'w_b' has no name"):
        figures.forest_rows("rsa", _domain_data("rsa"), {"w": "Winner one", "s1": "x", "s2": "y"},
                            PAPERS)


def test_a_starting_model_not_in_seed_papers_raises():
    with pytest.raises(KeyError, match="'s2' is not in seed_papers.yaml"):
        figures.forest_rows("rsa", _domain_data("rsa"), NAMES, {"s1": "Author, 1999"})


def test_starting_models_are_named_by_their_paper_and_the_best_by_its_name_without_one():
    def seed(model, diff, paper):
        return {"kind": "seed", "model": model, "label": model.upper(), "diff": diff, "paper": paper}

    columns = [seed("a", -9.0, "Frank & Goodman, 2012"), seed("b", -5.0, "Frank & Goodman, 2012"),
               seed("c", -2.0, None), seed("d", 0.0, None)]
    # One name for a paper's models; uncited models unnamed but the best.
    assert figures._seed_names(columns) == {"a": "Frank & Goodman (2012)",
                                            "b": "Frank & Goodman (2012)", "d": "D"}
    rows = figures.forest_rows("rsa", _domain_data("rsa"), NAMES, PAPERS)
    assert [figures._row_label(r) for r in rows if r["kind"] == "seed"] == [
        "Seed one\n(Author & Other, 1999)", "Seed two"]


def _ng_with_strip():
    data = _domain_data("number_game")
    data["pooled"].update(runs=["r1", "r2"], others_at_end={"r1__x": 15.0}, pruned_during_run={"r2__y": -3.0},
                          pruned_order={"r2__y": 4}, others_order={"r1__x": 7})
    return data


def _unseen(strip=True):
    def test_set(best_seed, shift):
        record = {"best_seed": best_seed, "n_responses": 1000, "models": {
            m: {"lpd": lpd + shift, "vs_best_seed": {"diff": diff, "se": se}}
            for m, (lpd, diff, se) in {"w": (-50.0, 30.0, 5.0), "w_b": (-60.0, 20.0, 4.0),
                                       "s1": (-90.0, -10.0, 3.0), "s2": (-80.0, 0.0, 0.0)}.items()}}
        if strip:
            record["strip"] = {"r1__x": 12.0, "r2__y": -7.0}
        return record
    return {"winner_runs": {"w": "r1", "w_b": "r2"}, "unseen_items": test_set("s2", 0.0),
            "unseen_example_sets": test_set("s2", 40.0)}


def test_figure_2s_number_game_is_its_models_scored_on_unseen_items():
    data = _ng_with_strip()
    view = figures.number_game_unseen_view(data, _unseen(), "unseen_items")
    rows = figures.forest_rows("number_game", view, NAMES, PAPERS)
    assert _labelled(rows) == [("winner", "Winner one", 30.0, 5.0), ("winner", "Winner two", 20.0, 4.0),
                               ("seed", "Seed two", 0.0, 0.0), ("seed", "Seed one", -10.0, 3.0)]
    (strip,) = [r for r in rows if "values" in r]
    assert strip["values"] == [12.0, -7.0]
    assert figures.best_seed_elpd("number_game", view) == -80.0
    assert figures.best_seed_elpd("number_game", figures.number_game_unseen_view(
        data, _unseen(), "unseen_example_sets")) == -40.0
    # Figures 3 and 4 keep the cross-validated comparison.
    assert data["pooled"]["best_seed"] == "s1" and data["pooled"]["others_at_end"] == {"r1__x": 15.0}


def test_an_unscored_unseen_strip_is_drawn_pending():
    view = figures.number_game_unseen_view(_ng_with_strip(), _unseen(strip=False), "unseen_items")
    (strip,) = [r for r in figures.forest_rows("number_game", view, NAMES, PAPERS) if "values" in r]
    assert strip["values"] is None


def test_unseen_scores_of_other_winners_raise():
    unseen = _unseen()
    unseen["winner_runs"] = {"w": "r1", "other": "r2"}
    with pytest.raises(RuntimeError, match="winners"):
        figures.number_game_unseen_view(_ng_with_strip(), unseen, "unseen_items")


def test_an_unseen_strip_missing_a_model_raises():
    unseen = _unseen()
    del unseen["unseen_items"]["strip"]["r2__y"]
    with pytest.raises(RuntimeError, match="r2__y"):
        figures.number_game_unseen_view(_ng_with_strip(), unseen, "unseen_items")


def test_figure_2_draws_the_best_textbook_model_only_where_the_data_record_one():
    rsa = {**_domain_data("rsa"), "best_textbook": {"model": "s1", "diff": -10.0}}

    assert figures.textbook_line(rsa) == -10.0
    assert figures.textbook_line(_domain_data("subjective_randomness")) is None



def test_promoted_seeds_are_coloured_apart_from_the_seeds_that_predate_auto_psych():
    rsa = {**_domain_data("rsa"), "promoted_seeds": ["s1"]}
    rows = figures.forest_rows("rsa", rsa, NAMES, PAPERS)

    colours = {r["model"]: figures.seed_color(rsa, r["model"]) for r in rows if r["kind"] == "seed"}
    assert colours == {"s1": figures.COLOR_PROMOTED, "s2": figures.COLOR_SEED}
    # Each panel's zero line is its best starting model's colour.
    assert figures.zero_line_color("rsa", rsa) == figures.COLOR_PROMOTED
    sr = _domain_data("subjective_randomness")
    assert figures.zero_line_color("subjective_randomness", sr) == figures.COLOR_SEED


def test_the_number_game_is_titled_number_sequence_generalization():
    assert figures.panel_title("number_game", mock_domains=[]) == "Number sequence generalization"
