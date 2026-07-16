"""A round's candidate agents run concurrently; admission stays sequential.

Candidate agents are CLI subprocesses whose wall-clock dominates a round, and
they are independent given the round's shared context (critique + posterior) —
so they spawn in parallel. Admission (which mutates the manifest, uniquifies
names, and runs the novelty gate) still happens sequentially in candidate
order, keeping runs deterministic: earlier candidates win ties.
"""

from __future__ import annotations

import threading

import yaml

import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop


def _make_seed_models(tmp_path):
    seed_dir = tmp_path / "seed_models"
    seed_dir.mkdir()
    (seed_dir / "model_a.py").write_text("# stub model_a\n", encoding="utf-8")
    (seed_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": "model_a"}]}, sort_keys=False),
        encoding="utf-8",
    )
    return seed_dir


def _make_responses(tmp_path):
    responses = tmp_path / "responses.csv"
    responses.write_text("chose_left,n_a\n1,6\n0,6\n", encoding="utf-8")
    return responses


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
    monkeypatch.setattr(pymc_orchestrator, "log_likelihood", lambda *a, **k: -100.0)
    monkeypatch.setattr(
        pymc_orchestrator,
        "_min_prediction_rmse",
        lambda *a, **k: (None, float("inf")),
    )
    monkeypatch.setattr(
        pymc_orchestrator, "load_pymc_model", lambda name, models_dir: object()
    )


def _run(tmp_path, monkeypatch, spawn, **kwargs):
    _patch_loop_internals(monkeypatch)
    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", spawn)
    return run_pymc_inner_loop(
        responses_path=_make_responses(tmp_path),
        results_dir=tmp_path / "model_loop",
        seed_models_dir=_make_seed_models(tmp_path),
        max_iterations=1,
        candidate_count=3,
        enable_critique=False,
        fit_kwargs={},
        **kwargs,
    )


def test_candidate_agents_spawn_concurrently(tmp_path, monkeypatch):
    # Each fake agent blocks until ALL three have started: this only completes
    # if the spawns genuinely overlap. A sequential loop would deadlock (the
    # barrier times out and the test fails loudly).
    barrier = threading.Barrier(3, timeout=10)

    def blocking_spawn(candidate_dir, docs, **kwargs):
        barrier.wait()
        (candidate_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text("People use H.\n", encoding="utf-8")
        return True

    result = _run(tmp_path, monkeypatch, blocking_spawn)
    assert result["best_model"] == "model_a"


def test_admission_order_is_sequential_and_deterministic(tmp_path, monkeypatch):
    admitted = []
    real_admit = pymc_orchestrator._admit_candidate

    def recording_admit(candidate_file, models_dir, model_name, *a, **k):
        admitted.append(model_name)
        return real_admit(candidate_file, models_dir, model_name, *a, **k)

    def spawn(candidate_dir, docs, **kwargs):
        (candidate_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text("People use H.\n", encoding="utf-8")
        return True

    monkeypatch.setattr(pymc_orchestrator, "_admit_candidate", recording_admit)
    _run(tmp_path, monkeypatch, spawn)
    assert admitted == ["iter0_candidate0", "iter0_candidate1", "iter0_candidate2"]


def test_parallelism_one_runs_agents_sequentially(tmp_path, monkeypatch):
    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def counting_spawn(candidate_dir, docs, **kwargs):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        (candidate_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text("People use H.\n", encoding="utf-8")
        with lock:
            active["now"] -= 1
        return True

    _run(tmp_path, monkeypatch, counting_spawn, candidate_parallelism=1)
    assert active["max"] == 1
