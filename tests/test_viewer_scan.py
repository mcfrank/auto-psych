"""Unit tests for run discovery and scanning.

A run is any directory under the data root that holds experiments (or a bare
model loop), found at whatever depth it lives. Each run scans into the same
structured payload regardless of where it is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.viewer.scan import scan_index, scan_run, scan_run_experiment
from tests.viewer_fixtures import build_demo_tree


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return build_demo_tree(tmp_path / "data")


def test_find_runs_at_all_depths(data_root: Path):
    index = scan_index(data_root)
    by_path = {r.path: r for r in index.runs}
    assert set(by_path) == {
        "outer_loop/demo",
        "recovery/holdout_runs/cond_a",
        "recovery/confusion_runs/model_x",
        "thinkaloud",
    }
    assert by_path["outer_loop/demo"].kind == "experiments"
    assert by_path["outer_loop/demo"].n_experiments == 2
    assert by_path["recovery/confusion_runs/model_x"].kind == "loop"


def test_scan_run_units_and_figures(data_root: Path):
    run = scan_run(data_root, "outer_loop/demo")
    units = {e.unit: e for e in run.experiments}
    assert set(units) == {"experiment1", "smoke"}
    assert run.figures == ["analysis/loop_trajectory.png"]
    assert units["experiment1"].best_model == "bayesian_fair_coin"


def test_scan_run_experiment_full(data_root: Path):
    exp = scan_run_experiment(data_root, "outer_loop/demo", "experiment1")
    assert {m.name for m in exp.theory.models} >= {"equally_likely", "bayesian_fair_coin"}
    assert exp.design.n_stimuli == 2
    assert exp.data.n_participants == 2
    assert exp.best_model == "bayesian_fair_coin"
    cand = next(c for c in exp.model_loop.candidates if c.name == "iter0_candidate0")
    assert "alternation rate" in cand.hypothesis
    assert "\x1b" not in (cand.transcript or "")


def test_inner_loop_export_marked_not_a_theory_seed(data_root: Path):
    # `inner_loop_model` is the inner loop's winner, exported into cognitive_models/
    # AFTER the loop ran — it must not be presented as a theory-step seed.
    exp = scan_run_experiment(data_root, "outer_loop/demo", "experiment1")
    by_name = {m.name: m for m in exp.theory.models}
    assert by_name["inner_loop_model"].origin == "inner_loop"
    assert by_name["equally_likely"].origin == "seed"
    assert by_name["bayesian_fair_coin"].origin == "seed"


def test_scan_run_experiment_partial(data_root: Path):
    exp = scan_run_experiment(data_root, "outer_loop/demo", "smoke")
    assert exp.theory.models == []
    assert exp.model_loop is None
    assert exp.data.n_rows == 1


def test_critique_statistics_and_results(data_root: Path):
    exp = scan_run_experiment(data_root, "outer_loop/demo", "experiment1")
    crit = next(c for c in exp.critiques if c.iteration == 0)
    assert crit.model == "bayesian_fair_coin"
    assert crit.n_significant == 1
    by_name = {s.name: s for s in crit.stats}
    sig = by_name["alternation_rate_gap"]
    assert sig.significant is True
    assert sig.p_value_adjusted == pytest.approx(0.024)
    assert sig.has_result is True
    assert by_name["lonely_stat"].has_result is False  # file, no result
    assert by_name["results_only_stat"].has_result is True  # result, no file


def test_experiment_level_critique_without_model_loop(data_root: Path):
    exp = scan_run_experiment(data_root, "thinkaloud", "experiment1")
    assert exp.model_loop is None
    crit = next(c for c in exp.critiques if c.iteration is None)
    stat = next(s for s in crit.stats if s.name == "mean_solve_rate")
    assert "solve rate" in stat.description.lower()
    assert stat.has_result is False  # no ppc_results.json written


def test_nested_run_has_candidates(data_root: Path):
    exp = scan_run_experiment(data_root, "recovery/holdout_runs/cond_a", "experiment1")
    assert exp.model_loop is not None
    assert any(c.name == "iter0_candidate0" for c in exp.model_loop.candidates)


def test_bare_loop_run(data_root: Path):
    run = scan_run(data_root, "recovery/confusion_runs/model_x")
    assert run.experiments[0].kind == "loop"
    unit = run.experiments[0].unit
    exp = scan_run_experiment(data_root, "recovery/confusion_runs/model_x", unit)
    assert exp.model_loop is not None
    assert exp.best_model == "prototype_similarity"


def test_corrupt_json_fails_loudly(data_root: Path):
    bad = data_root / "outer_loop" / "demo" / "experiment1" / "model_loop" / "history.json"
    bad.write_text("{ not valid json")
    with pytest.raises(ValueError, match="history.json"):
        scan_run_experiment(data_root, "outer_loop/demo", "experiment1")


def test_missing_run_fails_loudly(data_root: Path):
    with pytest.raises(FileNotFoundError):
        scan_run(data_root, "outer_loop/nope")
