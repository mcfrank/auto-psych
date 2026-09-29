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

import src.pipelines.inner_loop.model_zoo as model_zoo
import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
import src.pipelines.inner_loop.scoring as scoring
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from tests.inner_loop_fixtures import write_responses, write_seed_models


def _patch_loop_internals(monkeypatch):
    posterior = {
        "posteriors": {"model_a": 1.0},
        "elpd_loo": {"model_a": -10.0},
        "n_trials": 2,
    }
    monkeypatch.setattr(
        scoring, "model_posterior", lambda *a, **k: posterior
    )
    # A rank table naming every live model: a six-slot round's refinement
    # briefs describe the incumbent's standing from it (model_a is rank 0).
    def fake_compare(responses_path, models_dir, **kwargs):
        data = yaml.safe_load((models_dir / "models_manifest.yaml").read_text())
        names = [e["name"] for e in data["models"]]
        return {
            n: {
                "rank": i,
                "elpd_loo": -10.0 - i,
                "elpd_diff": 0.0 if i == 0 else 1.0,
                "dse": 0.0 if i == 0 else 5.0,
                "weight": 1.0 if i == 0 else 0.0,
                "loo_unreliable": False,
            }
            for i, n in enumerate(names)
        }

    monkeypatch.setattr(scoring, "compare_table", fake_compare)
    # _prune_losers looks up compare_table in model_zoo's namespace:
    monkeypatch.setattr(model_zoo, "compare_table", fake_compare)
    # Functions looked up in model_zoo's namespace (called by _admit_candidate,
    # _drop_unfittable_models, etc. which now live in model_zoo):
    monkeypatch.setattr(
        model_zoo, "model_logp_is_finite", lambda *a, **k: (True, "")
    )
    # The stub fit is not a real trace: pass the convergence gate.
    monkeypatch.setattr(model_zoo, "convergence_problems_of", lambda fitted: [])
    monkeypatch.setattr(model_zoo, "fit_model", lambda *a, **k: object())
    # The experiment-start screen samples the whole set in one batch; no MCMC here.
    monkeypatch.setattr(model_zoo, "fit_models_to_cache", lambda names, *a, **k: {})
    monkeypatch.setattr(model_zoo, "log_likelihood", lambda *a, **k: -100.0)
    monkeypatch.setattr(
        model_zoo,
        "_min_prediction_rmse",
        lambda *a, **k: (None, float("inf")),
    )
    monkeypatch.setattr(
        model_zoo, "load_pymc_model", lambda name, models_dir: object()
    )


def _run(tmp_path, monkeypatch, spawn, *, candidate_count=3, **kwargs):
    _patch_loop_internals(monkeypatch)
    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", spawn)
    return run_pymc_inner_loop(
        responses_path=write_responses(tmp_path),
        results_dir=tmp_path / "model_loop",
        seed_models_dir=write_seed_models(tmp_path, ["model_a"]),
        max_iterations=1,
        candidate_count=candidate_count,
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
    real_admit = pymc_orchestrator._admit_candidate_with_reason

    def recording_admit(candidate_file, models_dir, model_name, *a, **k):
        admitted.append(model_name)
        return real_admit(candidate_file, models_dir, model_name, *a, **k)

    def spawn(candidate_dir, docs, **kwargs):
        (candidate_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text("People use H.\n", encoding="utf-8")
        return True

    monkeypatch.setattr(
        pymc_orchestrator, "_admit_candidate_with_reason", recording_admit
    )
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


def _write_candidate_files(candidate_dir):
    (candidate_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
    (candidate_dir / "hypothesis.md").write_text("People use H.\n", encoding="utf-8")


def test_six_candidate_agents_spawn_concurrently_by_default(tmp_path, monkeypatch):
    """The scaled round (six slots: three exploratory, two refining the
    incumbent, one agent-chosen) runs every slot's agent at once. The default
    parallelism is one worker per slot, so a barrier that opens only when all
    six spawns overlap must open; a smaller pool would deadlock on it and the
    barrier's timeout would fail the test loudly."""
    barrier = threading.Barrier(6, timeout=10)

    def blocking_spawn(candidate_dir, docs, **kwargs):
        barrier.wait()
        _write_candidate_files(candidate_dir)
        return True

    result = _run(tmp_path, monkeypatch, blocking_spawn, candidate_count=6)
    assert result["best_model"] == "model_a"
    assert barrier.broken is False


def test_parallelism_cap_bounds_a_six_slot_round(tmp_path, monkeypatch):
    """``candidate_parallelism=3`` on six slots runs exactly three agents at a
    time: a three-party barrier proves three overlap, and the three-thread
    pool means the count of concurrently running spawns never exceeds three."""
    barrier = threading.Barrier(3, timeout=10)
    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def capped_spawn(candidate_dir, docs, **kwargs):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        barrier.wait()
        _write_candidate_files(candidate_dir)
        with lock:
            active["now"] -= 1
        return True

    _run(tmp_path, monkeypatch, capped_spawn, candidate_count=6, candidate_parallelism=3)
    assert active["max"] == 3
