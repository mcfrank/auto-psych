"""Unit tests for per-step history tracking in the PyMC inner loop.

The loop scores the model set once after seeding and once after each candidate
round; every scoring step must be recorded to ``history.json`` (so a crashed
run keeps a partial trajectory) and returned under ``"history"``. MCMC and
agent spawning are stubbed — these tests cover only the bookkeeping.
"""

from __future__ import annotations

import json

import yaml

import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop


def _make_seed_models(tmp_path):
    seed_dir = tmp_path / "seed_models"
    seed_dir.mkdir()
    for name in ("model_a", "model_b"):
        (seed_dir / f"{name}.py").write_text(f"# stub {name}\n", encoding="utf-8")
    (seed_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": "model_a"}, {"name": "model_b"}]}, sort_keys=False
        ),
        encoding="utf-8",
    )
    return seed_dir


def _make_responses(tmp_path):
    responses = tmp_path / "responses.csv"
    responses.write_text("chose_left,n_a\n1,6\n0,6\n", encoding="utf-8")
    return responses


def _posterior(best, others):
    names = [best] + list(others)
    posteriors = {name: (0.7 if name == best else 0.3 / len(others)) for name in names}
    return {
        "posteriors": posteriors,
        "elpd_loo": {name: -10.0 - i for i, name in enumerate(names)},
        "n_trials": 2,
    }


def _patch_scoring(monkeypatch, posteriors_per_call):
    calls = {"n": 0}

    def fake_model_posterior(responses_path, models_dir, **kwargs):
        result = posteriors_per_call[calls["n"]]
        calls["n"] += 1
        return result

    monkeypatch.setattr(pymc_orchestrator, "model_posterior", fake_model_posterior)
    monkeypatch.setattr(
        pymc_orchestrator, "compare_table", lambda *args, **kwargs: {}
    )
    # Stub fittability so the fake stub seed models are not dropped/scored as
    # un-fittable (they are not real PyMC models).
    monkeypatch.setattr(
        pymc_orchestrator, "model_logp_is_finite", lambda *a, **k: (True, "")
    )
    # Candidate admission now ends with a real MCMC fit-gate; stub it so the fake
    # stub candidates (not real PyMC models) are admitted without sampling.
    monkeypatch.setattr(pymc_orchestrator, "fit_model", lambda *a, **k: object())


def _patch_candidates(monkeypatch):
    """Each spawned agent writes a candidate; validation is stubbed to accept."""

    def fake_spawn(candidate_dir, agent_timeout_sec, backend):
        (candidate_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text(
            "People use heuristic H.\n", encoding="utf-8"
        )
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", fake_spawn)
    monkeypatch.setattr(
        pymc_orchestrator, "load_pymc_model", lambda name, models_dir: object()
    )


def test_inner_loop_writes_history_entry_per_scoring_step(tmp_path, monkeypatch):
    _patch_scoring(
        monkeypatch,
        [
            _posterior("model_a", ["model_b"]),
            _posterior("iter0_candidate0", ["model_a", "model_b"]),
            _posterior("iter1_candidate0", ["model_a", "model_b", "iter0_candidate0"]),
        ],
    )
    _patch_candidates(monkeypatch)

    result = run_pymc_inner_loop(
        _make_responses(tmp_path),
        tmp_path / "results",
        seed_models_dir=_make_seed_models(tmp_path),
        max_iterations=2,
        candidate_count=1,
    )

    history = result["history"]
    assert [entry["step"] for entry in history] == [0, 1, 2]
    assert [entry["iteration"] for entry in history] == [None, 0, 1]
    assert [entry["best_model"] for entry in history] == [
        "model_a",
        "iter0_candidate0",
        "iter1_candidate0",
    ]
    for entry in history:
        assert set(entry["posteriors"]) == set(entry["elpd_loo"])

    history_path = tmp_path / "results" / "history.json"
    assert result["history_path"] == str(history_path)
    assert json.loads(history_path.read_text(encoding="utf-8")) == history


def test_inner_loop_history_seed_only_has_single_step(tmp_path, monkeypatch):
    _patch_scoring(monkeypatch, [_posterior("model_b", ["model_a"])])

    result = run_pymc_inner_loop(
        _make_responses(tmp_path),
        tmp_path / "results",
        seed_models_dir=_make_seed_models(tmp_path),
        max_iterations=0,
    )

    history = json.loads(
        (tmp_path / "results" / "history.json").read_text(encoding="utf-8")
    )
    assert history == result["history"]
    assert len(history) == 1
    assert history[0]["step"] == 0
    assert history[0]["iteration"] is None
    assert history[0]["best_model"] == "model_b"
