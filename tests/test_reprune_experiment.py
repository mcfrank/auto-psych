"""Re-pruning a finished experiment at another multiplier (``outer_loop.reprune``).

Pruning runs once, at the end of an experiment's model loop, and nothing
before it depends on the multiplier, so redoing that step gives the set a
loop run with the new multiplier would have carried. At the same multiplier it
must leave the experiment exactly as it was; at a larger one it carries the
models within it into the export, the registry and the completion record. In
the October 2026 live run pruning at 2·dse_clustered left experiment 1 a
single model (user decision 2026-09-30: the series moved to 4).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import src.pipelines.outer_loop.run as outer_run
from src.models.model_manifest import read_manifest_names
from src.models.pymc_inference import cached_fit_path, fit_fingerprint, resolve_fit_settings
from src.pipelines.inner_loop import model_zoo, pymc_orchestrator, scoring
from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
from src.pipelines.outer_loop import model_loop_runner as mlr
from src.pipelines.outer_loop.orchestrator_validators import validate_cc_output
from src.pipelines.outer_loop.reprune import reprune_experiment
from src.registry.io import get_model_weights
from tests.paths import PYMC_MODEL_FIXTURES_DIR
from tests.test_parallel_candidates import _patch_loop_internals

PROJECT = "subjective_randomness"
FIT = {"draws": 10, "tune": 10, "chains": 1}
# Behind the best by this many clustered standard errors (dse_clustered = 1).
MARGINS = {"model_a": 0.0, "near": 3.0, "far": 5.0}
_SOURCE = (PYMC_MODEL_FIXTURES_DIR / "bayesian_fair_coin.py").read_text()


def _graded_compare(responses_path, models_dir, *, names=None, **kwargs):
    present = names or read_manifest_names(models_dir)
    ranked = sorted(present, key=MARGINS.__getitem__)
    return {
        name: {
            "rank": rank,
            "elpd_loo": -100.0 - MARGINS[name],
            "elpd_diff": MARGINS[name],
            "dse": 0.5 if rank else 0.0,
            "dse_clustered": 1.0 if rank else 0.0,
            "weight": 1.0 if rank == 0 else 0.0,
            "loo_unreliable": False,
        }
        for rank, name in enumerate(ranked)
    }


@pytest.fixture
def stubbed_loop(monkeypatch):
    _patch_loop_internals(monkeypatch)
    monkeypatch.setattr(scoring, "compare_table", _graded_compare)
    monkeypatch.setattr(model_zoo, "compare_table", _graded_compare)
    monkeypatch.setattr(pymc_orchestrator, "prefit_candidates", lambda *a, **k: [])

    def spawn(candidate_dir, docs, **kwargs):
        name = "near" if candidate_dir.name == "candidate_0" else "far"
        (candidate_dir / "candidate.py").write_text(f"# {name}\n{_SOURCE}", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text(
            f"People judge randomness by the {name} rule.\n", encoding="utf-8"
        )
        (candidate_dir / "model_name.txt").write_text(name, encoding="utf-8")
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", spawn)


def _finished_experiment(tmp_path: Path) -> Path:
    """Experiment 1 after its model loop, pruned at the default 2·dse."""
    exp_dir = tmp_path / "output" / PROJECT / "experiment1"
    (exp_dir / "data").mkdir(parents=True)
    (exp_dir / "data" / "responses.csv").write_text(
        "sequence_a,sequence_b,participant_id,trial_index,chose_left\n"
        "HHTTHT,HTHTHT,1,0,1\nHHTTHT,HTHTHT,2,0,0\n",
        encoding="utf-8",
    )
    models = exp_dir / "cognitive_models"
    models.mkdir()
    (models / "model_a.py").write_text(_SOURCE, encoding="utf-8")
    (models / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": "model_a", "rationale": "The seed."}]}),
        encoding="utf-8",
    )
    (models / "attempted_hypotheses.jsonl").write_text("", encoding="utf-8")
    mlr.init_registry(exp_dir)
    mlr.agent_notes_dir(exp_dir).mkdir()
    outer_run._run_agent(
        "5_model_loop", exp_dir, PROJECT, 1, "simulated_participants_nobrowser", 2,
        None, False, inner_loop_iterations=1, inner_loop_candidates=2,
        fit_kwargs=dict(FIT), enable_critique=False,
    )
    return exp_dir


def _cache_every_fit(exp_dir: Path, skip=()):
    """Stand-in cache entries for the loop's fits (the comparison is stubbed)."""
    loop = exp_dir / "model_loop"
    for directory in (loop / "models", loop / "models" / "pruned"):
        for model_file in directory.glob("*.py"):
            name = model_file.stem
            if name in skip:
                continue
            settings = resolve_fit_settings(name, directory, FIT)
            path = cached_fit_path(
                loop / ".fit_cache", name,
                fit_fingerprint(name, directory, loop / "responses.csv", settings),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


def _state(exp_dir: Path) -> dict:
    loop = exp_dir / "model_loop"
    files = [
        loop / "models" / "models_manifest.yaml",
        loop / "attempted_hypotheses.jsonl",
        loop / "history.json",
        loop / "export_complete.json",
        exp_dir / "cognitive_models" / "models_manifest.yaml",
        exp_dir / "cognitive_models" / "attempted_hypotheses.jsonl",
        exp_dir / "model_registry.yaml",
    ]
    state = {str(p.relative_to(exp_dir)): p.read_text(encoding="utf-8") for p in files}
    state["zoo files"] = sorted(p.name for p in (loop / "models").iterdir())
    state["pruned files"] = sorted(p.name for p in (loop / "models" / "pruned").iterdir())
    state["carried files"] = sorted(p.name for p in (exp_dir / "cognitive_models").iterdir())
    return state


def test_the_stub_loop_pruned_both_candidates_at_two_dse(tmp_path, stubbed_loop):
    exp_dir = _finished_experiment(tmp_path)
    assert read_manifest_names(exp_dir / "cognitive_models") == ["model_a"]


def test_reprune_at_the_same_multiplier_leaves_the_experiment_as_it_was(
    tmp_path, stubbed_loop
):
    exp_dir = _finished_experiment(tmp_path)
    _cache_every_fit(exp_dir)
    before = _state(exp_dir)

    reprune_experiment(exp_dir, dse_multiplier=2.0, fit_kwargs=dict(FIT))

    assert _state(exp_dir) == before


def test_reprune_at_four_dse_carries_the_model_within_four(tmp_path, stubbed_loop):
    exp_dir = _finished_experiment(tmp_path)
    _cache_every_fit(exp_dir)

    summary = reprune_experiment(
        exp_dir, dse_multiplier=4.0, fit_kwargs=dict(FIT), note="test"
    )

    assert summary["restored"] == ["near"] and summary["retired"] == ["far"]
    assert read_manifest_names(exp_dir / "model_loop" / "models") == ["model_a", "near"]
    assert read_manifest_names(exp_dir / "cognitive_models") == ["model_a", "near"]
    assert (exp_dir / "cognitive_models" / "near.py").exists()
    assert (exp_dir / "model_loop" / "models" / "pruned" / "far.py").exists()
    ledger = HypothesisLedger(exp_dir / "model_loop" / "attempted_hypotheses.jsonl")
    pruned = [(e.name, e.context) for e in ledger.entries() if e.outcome == "pruned"]
    assert pruned == [("far", "experiment1 end of experiment")]
    weights = get_model_weights(exp_dir / "model_registry.yaml")
    assert weights == pytest.approx({"model_a": 0.5, "near": 0.5})
    assert validate_cc_output("5_model_loop", exp_dir)[0]
    record = json.loads((exp_dir / "model_loop" / "repruned.json").read_text())
    assert record[-1]["dse_multiplier"] == 4.0 and record[-1]["note"] == "test"


def test_reprune_refuses_once_the_next_experiment_exists(tmp_path, stubbed_loop):
    exp_dir = _finished_experiment(tmp_path)
    _cache_every_fit(exp_dir)
    (exp_dir.parent / "experiment2").mkdir()
    before = _state(exp_dir)

    with pytest.raises(RuntimeError, match="experiment2"):
        reprune_experiment(exp_dir, dse_multiplier=4.0, fit_kwargs=dict(FIT))
    assert _state(exp_dir) == before


def test_reprune_refuses_a_fit_it_would_have_to_sample(tmp_path, stubbed_loop):
    exp_dir = _finished_experiment(tmp_path)
    _cache_every_fit(exp_dir, skip={"far"})
    before = _state(exp_dir)

    with pytest.raises(RuntimeError, match="No cached fit of 'far'"):
        reprune_experiment(exp_dir, dse_multiplier=4.0, fit_kwargs=dict(FIT))
    assert _state(exp_dir) == before
