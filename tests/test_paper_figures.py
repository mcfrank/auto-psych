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
        (data_dir / f"{domain}.json").write_text(json.dumps(_domain_data(domain)))
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
    return figures.Args(figure_data=data_dir, pseudocode=spec_dir, out_dir=tmp_path / "out",
                        model_names=names, pending_domains=["number_game"])


NAMES = {"w": "Winner one", "w_b": "Winner two", "s1": "Seed one", "s2": "Seed two"}


def test_all_three_figures_are_drawn_with_the_number_game_pending(tmp_path):
    args = _setup(tmp_path)
    figures.main(args)
    for name in ("fig2_elpd", "fig3_conditions", "fig4_pseudocode"):
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
    rows = figures.forest_rows("rsa", _domain_data("rsa"), NAMES)
    # Winners first (best first, with their run), then the strips, then the seeds.
    assert _labelled(rows) == [
        ("winner", "Winner two (r2)", 13.0, 3.0),
        ("winner", "Winner one (r1)", 12.0, 3.0),
        ("seed", "Seed one", 0.0, 0.0),
        ("seed", "Seed two", -8.0, None),  # no paired SE recorded: drawn hollow
    ]
    strips = {r["kind"]: r["values"] for r in rows if "values" in r}
    assert strips == {"others": None, "pruned": None}  # pruned not scored: both pending


def test_pruned_models_of_every_run_share_one_strip():
    data = _domain_data("number_game")
    for k, run in enumerate(data["runs"]):
        run["pruned_during_run"] = {f"late_{k}": 4.0 - k, f"early_{k}": -k}
        run["pruned_order"] = {f"late_{k}": 9, f"early_{k}": 2}
    rows = figures.forest_rows("number_game", data, NAMES)
    strips = {r["kind"]: r["values"] for r in rows if "values" in r}
    assert strips == {"others": [9.0, 10.0], "pruned": [0.0, 4.0, -1.0, 3.0]}
    # Each run's models in admission order, from 0 (first) to 1 (last).
    assert next(r for r in rows if r["kind"] == "pruned")["progress"] == [0.0, 1.0, 0.0, 1.0]


def test_subjective_randomness_is_drawn_from_all_runs_pooled():
    data = _domain_data("subjective_randomness")
    rows = figures.forest_rows("subjective_randomness", data, NAMES)
    # Before the pooled paired SEs exist: differences from the pooled ELPDs.
    assert _labelled(rows) == [("winner", "Winner one (r1)", 20.0, None),
                               ("winner", "Winner two (r2)", 5.0, None),
                               ("seed", "Seed one", 0.0, 0.0), ("seed", "Seed two", -30.0, None)]
    data["pooled"]["vs_best_seed"] = {"w": {"diff": 20.0, "se": 4.0}, "w_b": {"diff": 5.0, "se": 2.0},
                                      "s1": {"diff": 0.0, "se": 0.0}, "s2": {"diff": -30.0, "se": 6.0}}
    data["pooled"].update(runs=["r1", "r2"], others_at_end={"r1__x": 15.0},
                          pruned_during_run={"r2__y": -3.0}, pruned_order={"r2__y": 4})
    rows = figures.forest_rows("subjective_randomness", data, NAMES)
    assert [r["se"] for r in rows if "se" in r] == [4.0, 2.0, 0.0, 6.0]
    assert {r["kind"]: r["values"] for r in rows if "values" in r} == {"others": [15.0],
                                                                         "pruned": [-3.0]}


def test_a_labelled_model_without_a_name_raises():
    with pytest.raises(KeyError, match="'w_b' has no name"):
        figures.forest_rows("rsa", _domain_data("rsa"), {"w": "Winner one", "s1": "x", "s2": "y"})
