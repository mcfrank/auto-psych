"""Every exploration lens fires over the course of an outer-loop run.

The brief's lens was ``hints[candidate_idx % len(hints)]``: with three
candidates per round only lenses 0-2 ever fired (72 briefs each in the
iteration-4 recovery sweep, none for lenses 3-6). The lens now walks the
battery across rounds *and* experiments: slot
``lens_offset + iteration * candidate_count + idx`` modulo the battery size,
where the outer loop sets ``lens_offset`` from the experiment number so
experiment k+1 continues where experiment k stopped.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest
import yaml

import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
import src.pipelines.outer_loop.orchestrator as orch
from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME
from src.pipelines.inner_loop.pymc_orchestrator import (
    DEFAULT_CANDIDATE_HINTS,
    _lens_index,
    _lens_offset,
    run_pymc_inner_loop,
)
from tests.inner_loop_fixtures import write_responses, write_seed_models


# ── (a) Pure schedule ──────────────────────────────────────────────────

def test_lens_index_formula():
    assert _lens_index(0, 0, 3, 0, 7) == 0
    assert _lens_index(0, 0, 3, 1, 7) == 1
    assert _lens_index(0, 0, 3, 2, 7) == 2
    assert _lens_index(0, 1, 3, 0, 7) == 3
    assert _lens_index(6, 0, 3, 0, 7) == 6
    assert _lens_index(6, 0, 3, 1, 7) == 0


def test_lens_schedule_covers_the_battery_across_experiments():
    n = len(DEFAULT_CANDIDATE_HINTS)
    fired = Counter()
    for exp_num in (1, 2, 3):
        offset = _lens_offset(exp_num, max_iterations=2, candidate_count=3)
        for iteration in range(2):
            for idx in range(3):
                fired[_lens_index(offset, iteration, 3, idx, n)] += 1
    assert set(fired) == set(range(n))
    assert max(fired.values()) - min(fired.values()) <= 1
    assert _lens_offset(2, max_iterations=2, candidate_count=3) == 6
    assert _lens_index(6, 0, 3, 0, n) == 6
    assert _lens_index(6, 0, 3, 1, n) == 0


def test_empty_lens_battery_raises():
    with pytest.raises(ValueError, match="empty"):
        _lens_index(0, 0, 3, 0, 0)


def test_lens_offset_experiment_one_is_zero():
    assert _lens_offset(1, max_iterations=2, candidate_count=3) == 0


def test_lens_offset_rejects_zero():
    with pytest.raises(ValueError, match="start at 1"):
        _lens_offset(0, max_iterations=2, candidate_count=3)


# ── (b) Lens text in CANDIDATE_BRIEF.md matches ledger ────────────────

def _patch_loop_internals(monkeypatch):
    posterior = {
        "posteriors": {"model_a": 1.0},
        "elpd_loo": {"model_a": -10.0},
        "n_trials": 2,
    }
    monkeypatch.setattr(
        pymc_orchestrator, "model_posterior", lambda *a, **k: posterior
    )
    monkeypatch.setattr(pymc_orchestrator, "compare_table", lambda *a, **k: {})
    monkeypatch.setattr(
        pymc_orchestrator, "model_logp_is_finite", lambda *a, **k: (True, "")
    )
    monkeypatch.setattr(pymc_orchestrator, "fit_model", lambda *a, **k: object())
    monkeypatch.setattr(
        pymc_orchestrator, "log_likelihood", lambda *a, **k: -100.0
    )
    monkeypatch.setattr(
        pymc_orchestrator,
        "_min_prediction_rmse",
        lambda *a, **k: (None, float("inf")),
    )
    monkeypatch.setattr(
        pymc_orchestrator, "load_pymc_model", lambda n, d: object()
    )
    monkeypatch.setattr(
        pymc_orchestrator, "evict_fit_cache", lambda name: None
    )


def test_brief_lens_matches_ledger_lens(tmp_path, monkeypatch):
    """The lens text in a candidate's CANDIDATE_BRIEF.md matches the lens
    index the ledger records for that slot."""
    _patch_loop_internals(monkeypatch)
    briefs = {}

    def fake_spawn(candidate_dir, docs, **kwargs):
        rnd = int(candidate_dir.parent.name.split("_")[1])
        idx = int(candidate_dir.name.split("_")[1])
        briefs[(rnd, idx)] = docs["brief"]
        (candidate_dir / "candidate.py").write_text(
            "# candidate\n", encoding="utf-8"
        )
        (candidate_dir / "hypothesis.md").write_text(
            "People use H.\n", encoding="utf-8"
        )
        (candidate_dir / "model_name.txt").write_text(
            f"idea_{rnd}_{idx}\n", encoding="utf-8"
        )
        return True

    monkeypatch.setattr(
        pymc_orchestrator, "_spawn_candidate_agent", fake_spawn
    )
    run_pymc_inner_loop(
        responses_path=write_responses(tmp_path),
        results_dir=tmp_path / "model_loop",
        seed_models_dir=write_seed_models(tmp_path, ["model_a"]),
        max_iterations=2,
        candidate_count=3,
        enable_critique=False,
        fit_kwargs={},
        lens_offset=6,
        ledger_context="experiment2",
    )

    n = len(DEFAULT_CANDIDATE_HINTS)
    for (rnd, idx), brief in briefs.items():
        expected_lens = (6 + rnd * 3 + idx) % n
        expected_text = DEFAULT_CANDIDATE_HINTS[expected_lens]
        assert expected_text in brief, (rnd, idx, expected_lens)
        others = [h for h in DEFAULT_CANDIDATE_HINTS if h != expected_text]
        assert not any(h in brief for h in others), (rnd, idx)

    ledger = tmp_path / "model_loop" / LEDGER_FILENAME
    contexts = {
        r["name"]: r["context"]
        for r in (json.loads(l) for l in ledger.read_text().splitlines())
        if r["outcome"] == "admitted"
    }
    assert contexts["idea_0_0"] == "experiment2 round 0 candidate 0 lens 6"
    assert contexts["idea_0_1"] == "experiment2 round 0 candidate 1 lens 0"
    assert contexts["idea_1_2"] == "experiment2 round 1 candidate 2 lens 4"


# ── (c) Outer loop threads lens_offset from experiment number ──────────

def _outer_wrapper_capture(tmp_path, monkeypatch, exp_dir):
    exp_dir.mkdir(parents=True)
    (exp_dir / "cognitive_models").mkdir()
    (exp_dir / "cognitive_models" / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": "falk_konold_dp", "rationale": "seed"}]}
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_inner_loop(responses_path, results_dir, **inner_kwargs):
        captured.update(inner_kwargs)
        return {"best_model": "stub_best"}

    monkeypatch.setattr(
        orch, "_pooled_response_rows", lambda e: [{"chose_left": "1"}]
    )
    monkeypatch.setattr(
        orch, "_load_project_featurizer", lambda project_dir: None
    )
    monkeypatch.setattr(
        orch, "_write_feature_csv", lambda rows, fz, out: out
    )
    monkeypatch.setattr(
        orch,
        "_export_inner_loop_models",
        lambda e, l, *, best_model, protected_names: e,
    )
    monkeypatch.setattr(
        "src.pipelines.inner_loop.pymc_orchestrator.run_pymc_inner_loop",
        fake_inner_loop,
    )
    return captured


def test_outer_loop_sets_lens_offset_from_experiment_number(
    tmp_path, monkeypatch
):
    exp_dir = tmp_path / "subjective_randomness" / "experiment3"
    captured = _outer_wrapper_capture(tmp_path, monkeypatch, exp_dir)
    orch.run_inner_model_loop_programmatic(
        exp_dir,
        max_iterations=2,
        candidate_count=3,
        project_id="subjective_randomness",
    )
    assert captured["lens_offset"] == 12


def test_outer_loop_refuses_experiment_dir_without_a_number(
    tmp_path, monkeypatch
):
    exp_dir = tmp_path / "subjective_randomness" / "pilot"
    _outer_wrapper_capture(tmp_path, monkeypatch, exp_dir)
    with pytest.raises(ValueError, match="experiment<k>"):
        orch.run_inner_model_loop_programmatic(
            exp_dir,
            max_iterations=2,
            candidate_count=3,
            project_id="subjective_randomness",
        )


# ── (d) Empty lens battery raises ─────────────────────────────────────

def test_run_pymc_inner_loop_empty_lens_battery_raises(tmp_path, monkeypatch):
    _patch_loop_internals(monkeypatch)
    monkeypatch.setattr(
        pymc_orchestrator, "_spawn_candidate_agent", lambda *a, **k: True
    )
    with pytest.raises(ValueError, match="[Ee]mpty"):
        run_pymc_inner_loop(
            responses_path=write_responses(tmp_path),
            results_dir=tmp_path / "model_loop",
            seed_models_dir=write_seed_models(tmp_path, ["model_a"]),
            max_iterations=1,
            candidate_count=1,
            enable_critique=False,
            fit_kwargs={},
            candidate_hints=[],
        )
