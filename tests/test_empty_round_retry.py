"""An empty candidate round retries before abandoning, and only an experiment
whose every round is empty aborts the cell.

Background: in sweep 2, ~3% of three-slot rounds produced no files at all (the
opencode write permission fix succeeds ~70% per slot via bash heredocs). P17's
``AllCandidatesNoFileError`` was correct to refuse to continue silently, but
too blunt: a transient failure in one round killed cells that had already
admitted 5–8 candidates. The bounded retry keeps the safety guarantee (a
systemic bug still raises) while tolerating the expected empty-round rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import src.pipelines.inner_loop.model_zoo as model_zoo
import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
import src.pipelines.inner_loop.scoring as scoring
from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME
from src.pipelines.inner_loop.model_zoo import (
    MAX_EMPTY_ROUND_RETRIES,
    AllCandidatesNoFileError,
    _is_all_no_file_round,
)
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from tests.inner_loop_fixtures import write_responses, write_seed_models


def _manifest_names(models_dir):
    data = yaml.safe_load((models_dir / "models_manifest.yaml").read_text())
    return [e["name"] for e in data["models"]]


def _patch_scoring(monkeypatch):
    """Model_a always wins; everything else within noise (never pruned)."""

    def fake_model_posterior(responses_path, models_dir, **kwargs):
        names = _manifest_names(models_dir)
        return {
            "posteriors": {n: (1.0 if n == "model_a" else 0.0) for n in names},
            "elpd_loo": {n: (-10.0 if n == "model_a" else -11.0) for n in names},
            "n_trials": 2,
        }

    def fake_compare(responses_path, models_dir, **kwargs):
        names = _manifest_names(models_dir)
        rows = {}
        for rank, n in enumerate(["model_a"] + [m for m in names if m != "model_a"]):
            if n not in names:
                continue
            rows[n] = {
                "rank": rank,
                "elpd_loo": -10.0 - 1.0 * rank,
                "elpd_diff": 0.0 if n == "model_a" else 1.0,
                "dse": 0.0 if n == "model_a" else 5.0,
                "weight": 1.0 if n == "model_a" else 0.0,
                "loo_unreliable": False,
            }
        return rows

    monkeypatch.setattr(scoring, "model_posterior", fake_model_posterior)
    monkeypatch.setattr(scoring, "compare_table", fake_compare)
    monkeypatch.setattr(model_zoo, "compare_table", fake_compare)
    monkeypatch.setattr(
        model_zoo, "model_logp_is_finite", lambda *a, **k: (True, "")
    )
    monkeypatch.setattr(model_zoo, "fit_model", lambda *a, **k: object())
    monkeypatch.setattr(model_zoo, "log_likelihood", lambda *a, **k: -100.0)
    monkeypatch.setattr(model_zoo, "evict_fit_cache", lambda name: None)
    monkeypatch.setattr(
        model_zoo, "load_pymc_model", lambda name, models_dir: object()
    )
    monkeypatch.setattr(
        model_zoo,
        "_min_prediction_rmse",
        lambda name, *a, **k: (None, float("inf")),
    )


def _ledger_rows(path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]


# ── Unit tests for _is_all_no_file_round ────────────────────────────────


class TestIsAllNoFileRound:
    def test_all_no_file_returns_true(self):
        results = [
            {"outcome": "rejected", "detail": "no candidate.py written"},
            {"outcome": "rejected", "detail": "no candidate.py written"},
        ]
        assert _is_all_no_file_round(results) is True

    def test_some_admitted_returns_false(self):
        results = [
            {"outcome": "admitted", "detail": ""},
            {"outcome": "rejected", "detail": "no candidate.py written"},
        ]
        assert _is_all_no_file_round(results) is False

    def test_rejected_for_other_reason_returns_false(self):
        results = [
            {"outcome": "rejected", "detail": "no candidate.py written"},
            {"outcome": "rejected", "detail": "not a loadable PyMC model: ..."},
        ]
        assert _is_all_no_file_round(results) is False

    def test_spawn_failures_counted_as_no_file(self):
        results = [
            {"outcome": "spawn_failed", "detail": "agent process failed"},
            {"outcome": "rejected", "detail": "no candidate.py written"},
        ]
        assert _is_all_no_file_round(results) is True

    def test_empty_round_returns_false(self):
        assert _is_all_no_file_round([]) is False


# ── Integration tests for retry and experiment-level abort ──────────────


def test_round_fails_then_succeeds_on_retry(tmp_path, monkeypatch):
    """First attempt: all candidates produce no file. Retry: one admitted."""
    seed_dir = write_seed_models(tmp_path)
    responses = write_responses(tmp_path)
    _patch_scoring(monkeypatch)

    def fake_spawn(candidate_dir, docs, **kwargs):
        round_dir_name = candidate_dir.parent.name
        is_retry_dir = "_retry_" in round_dir_name
        if is_retry_dir:
            (candidate_dir / "candidate.py").write_text(
                "# ok\n", encoding="utf-8"
            )
            (candidate_dir / "hypothesis.md").write_text(
                "test hyp\n", encoding="utf-8"
            )
            (candidate_dir / "model_name.txt").write_text(
                "good_model\n", encoding="utf-8"
            )
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", fake_spawn)
    results_dir = tmp_path / "model_loop"

    run_pymc_inner_loop(
        responses,
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=1,
        candidate_count=1,
        enable_critique=False,
    )

    assert "good_model" in _manifest_names(results_dir / "models")
    rows = _ledger_rows(results_dir / LEDGER_FILENAME)
    admitted = [r for r in rows if r["outcome"] == "admitted"]
    assert len(admitted) == 1
    assert admitted[0]["name"] == "good_model"


def test_round_fails_twice_is_abandoned_loop_continues(tmp_path, monkeypatch):
    """Both attempts produce no files; round is abandoned, loop continues to
    the next round which succeeds."""
    seed_dir = write_seed_models(tmp_path)
    responses = write_responses(tmp_path)
    _patch_scoring(monkeypatch)

    def fake_spawn(candidate_dir, docs, **kwargs):
        round_dir_name = candidate_dir.parent.name
        if round_dir_name == "iter_1":
            (candidate_dir / "candidate.py").write_text(
                "# ok\n", encoding="utf-8"
            )
            (candidate_dir / "hypothesis.md").write_text(
                "test hyp\n", encoding="utf-8"
            )
            (candidate_dir / "model_name.txt").write_text(
                "round1_model\n", encoding="utf-8"
            )
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", fake_spawn)
    results_dir = tmp_path / "model_loop"

    run_pymc_inner_loop(
        responses,
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=2,
        candidate_count=1,
        enable_critique=False,
    )

    rows = _ledger_rows(results_dir / LEDGER_FILENAME)
    abandoned = [r for r in rows if r["outcome"] == "round_abandoned"]
    assert len(abandoned) == 1
    assert "round 0" in abandoned[0]["context"]
    admitted = [r for r in rows if r["outcome"] == "admitted"]
    assert len(admitted) == 1
    assert admitted[0]["name"] == "round1_model"


def test_all_rounds_fail_raises(tmp_path, monkeypatch):
    """Every round of the experiment fails on all retries => raises."""
    seed_dir = write_seed_models(tmp_path)
    responses = write_responses(tmp_path)
    _patch_scoring(monkeypatch)

    def fake_spawn(candidate_dir, docs, **kwargs):
        return True  # Spawn succeeds but never writes files

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", fake_spawn)
    results_dir = tmp_path / "model_loop"

    with pytest.raises(AllCandidatesNoFileError, match="Every round"):
        run_pymc_inner_loop(
            responses,
            results_dir,
            seed_models_dir=seed_dir,
            max_iterations=2,
            candidate_count=1,
            enable_critique=False,
        )


def test_ledger_records_retry_and_abandonment_with_context(tmp_path, monkeypatch):
    """The ledger records both the retry attempts and the abandonment with
    enough context for the verifier to count them."""
    seed_dir = write_seed_models(tmp_path)
    responses = write_responses(tmp_path)
    _patch_scoring(monkeypatch)

    spawn_calls = []

    def fake_spawn(candidate_dir, docs, **kwargs):
        spawn_calls.append(str(candidate_dir))
        round_dir_name = candidate_dir.parent.name
        is_round_1_first_attempt = round_dir_name == "iter_1"
        if is_round_1_first_attempt:
            (candidate_dir / "candidate.py").write_text(
                "# ok\n", encoding="utf-8"
            )
            (candidate_dir / "hypothesis.md").write_text(
                "a hypothesis\n", encoding="utf-8"
            )
            (candidate_dir / "model_name.txt").write_text(
                "new_idea\n", encoding="utf-8"
            )
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", fake_spawn)
    results_dir = tmp_path / "model_loop"

    run_pymc_inner_loop(
        responses,
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=2,
        candidate_count=1,
        enable_critique=False,
        ledger_context="experiment3",
    )

    rows = _ledger_rows(results_dir / LEDGER_FILENAME)
    abandoned = [r for r in rows if r["outcome"] == "round_abandoned"]
    assert len(abandoned) == 1
    assert "experiment3" in abandoned[0]["context"]
    assert "round 0" in abandoned[0]["context"]
    assert "abandoned after" in abandoned[0]["detail"].lower()
    assert "attempts" in abandoned[0]["detail"].lower()
