"""The RSA data behind the paper's Figures 2 and 3
(``scripts/paper/rsa_figure_data.py``), read from the campaign's aggregates:
the pre-registered test (``prereg_analysis.json``) and the per-display
predictions (``panels.json``). No fitting."""

from __future__ import annotations

import json

import pytest

from scripts.paper.rsa_figure_data import Args, main


def _prereg(best_promoted="promoted:p1"):
    return {
        "n_trials": 100,
        "finals": ["final_a", "final_b"],
        "best_starting": "seed:s1",
        "best_promoted": best_promoted,
        # The pre-registered test: each final against the best promoted seed.
        "primary": {
            "final_a": {"diff": 30.0, "se": 4.0, "z": 7.5},
            "final_b": {"diff": 10.0, "se": 3.0},
        },
        "lpd": {
            "final_a": -70.0, "final_b": -90.0,
            "seed:s1": -110.0, "seed:s2": -120.0,
            "promoted:p1": -100.0, "promoted:p2": -105.0, "promoted:p3": -108.0,
        },
    }


def _chains():
    """Each chain's starting models: its share of the promoted seeds."""
    return {"chains": [{"chain": 0, "models": ["p2", "p3"]}, {"chain": 1, "models": ["p1", "p3"]}]}


def _panels():
    return {
        "models": {
            "c0": {"model": "final_a"}, "c1": {"model": "final_b"},
            "promoted": {"model": "p1"},
        },
        "panels": [
            {
                "dataset": "prereg", "n": 10,
                "classes": [
                    {"cls": 0, "count": 7, "preds": {"c0": 0.6, "c1": 0.5, "promoted": 0.4}},
                    {"cls": 1, "count": 3, "preds": {"c0": 0.4, "c1": 0.5, "promoted": 0.6}},
                ],
            }
        ],
    }


def _run(tmp_path, prereg, panels, chains=None):
    (tmp_path / "prereg.json").write_text(json.dumps(prereg))
    (tmp_path / "panels.json").write_text(json.dumps(panels))
    (tmp_path / "chains.json").write_text(json.dumps(chains or _chains()))
    out = tmp_path / "rsa.json"
    main(Args(prereg=tmp_path / "prereg.json", panels=tmp_path / "panels.json",
              chain_seeds=tmp_path / "chains.json", chains=["c0", "c1"], out=out))
    return json.loads(out.read_text())


def test_each_chain_is_its_starting_models_and_final_against_the_best_promoted_seed(tmp_path):
    data = _run(tmp_path, _prereg(), _panels())

    assert data["domain"] == "rsa"
    assert [r["run"] for r in data["runs"]] == ["c0", "c1"]
    c0, c1 = data["runs"]
    # The zero is the pre-registered bar, the best promoted seed of all
    # chains (here not one of chain 0's own), and the final's SE is the test's.
    assert c0["best_model"] == "final_a"
    assert c0["best_seed"] == "p1"
    assert c0["n_trials"] == 100
    assert c0["vs_best_seed"] == {
        "final_a": {"diff": 30.0, "se": 4.0},
        "p2": {"diff": -5.0, "se": None},
        "p3": {"diff": -8.0, "se": None},
    }
    assert c1["vs_best_seed"]["p1"] == {"diff": 0.0, "se": 0.0}
    assert c0["others_at_end"] == {}
    # The textbook models (the campaign's "starting models") are not drawn.
    assert "shared_models" not in data
    # Figure 3: the best final by held-out density and the best promoted
    # seed, both fitted on the pooled data (panels.json).
    assert "best_run" not in data
    assert data["pooled"]["best_model"] == "final_a"
    assert data["pooled"]["best_seed"] == "p1"
    assert data["pooled"]["winner_runs"] == {"final_a": "c0", "final_b": "c1"}

    (cls0, cls1) = data["conditions"]
    assert cls0 == {"dataset": "prereg", "display": 0, "cls": 0, "n": 10,
                    "observed": 0.7, "best_model": 0.6, "best_seed": 0.4}
    assert cls1["observed"] == pytest.approx(0.3)


def test_a_recorded_difference_that_disagrees_with_the_totals_raises(tmp_path):
    prereg = _prereg()
    prereg["primary"]["final_a"]["diff"] = 31.0
    with pytest.raises(RuntimeError, match="totals"):
        _run(tmp_path, prereg, _panels())


def test_a_best_promoted_seed_that_is_not_the_best_by_density_raises(tmp_path):
    with pytest.raises(RuntimeError, match="best promoted seed"):
        _run(tmp_path, _prereg(best_promoted="promoted:p2"), _panels())


def test_panels_whose_promoted_seed_is_not_the_best_raise(tmp_path):
    panels = _panels()
    panels["models"]["promoted"]["model"] = "p2"
    with pytest.raises(RuntimeError, match="panels.json"):
        _run(tmp_path, _prereg(), panels)


def test_a_chain_starting_model_the_test_did_not_score_raises(tmp_path):
    chains = _chains()
    chains["chains"][0]["models"].append("p9")
    with pytest.raises(RuntimeError, match="p9"):
        _run(tmp_path, _prereg(), _panels(), chains)
