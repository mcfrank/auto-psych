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
    return {
        "domain": domain,
        "scoring": "ELPD on the final data",
        "runs": [
            {
                "run": run, "n_trials": 100, "best_model": winner, "best_seed": seed,
                "vs_best_seed": {
                    winner: {"diff": 12.0 + k, "se": 3.0},
                    seed: {"diff": 0.0, "se": 0.0},
                    "s2": {"diff": -8.0, "se": None},
                },
                "others_at_end": {"rival": 9.0 + k},
            }
            for k, run in enumerate(["r1", "r2"])
        ],
        "pooled": {"best_model": winner, "best_seed": seed},
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
    return figures.Args(figure_data=data_dir, pseudocode=spec_dir, out_dir=tmp_path / "out")


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


def test_each_run_is_a_column_of_its_seeds_its_other_models_and_its_winner():
    columns = figures.elpd_columns("subjective_randomness", _domain_data("subjective_randomness"))
    assert [c["label"] for c in columns] == ["r1", "r2"]  # the fixture's run names
    run1 = columns[0]
    assert run1["seeds"] == [0.0, -8.0]
    assert run1["others"] == [9.0]
    assert run1["winner"] == {"diff": 12.0, "se": 3.0}
