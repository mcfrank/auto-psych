"""Inner-loop integration tests for the CriticAL critique round.

Before each candidate-generation round the inner loop critiques the current
incumbent (best) model: it spawns a critique agent that posterior-predictively
checks the incumbent and writes ``critiques.md``, then passes that critique to
the candidate agents so they target the model's actual failure modes. MCMC and
both agent spawns are stubbed — these tests cover only the orchestration:

* a critique round runs before candidates each iteration,
* the candidate context points at the round's ``critiques.md``,
* ``enable_critique=False`` skips the critique entirely.
"""

from __future__ import annotations

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
    monkeypatch.setattr(pymc_orchestrator, "compare_table", lambda *a, **k: {})
    monkeypatch.setattr(
        pymc_orchestrator, "model_logp_is_finite", lambda *a, **k: (True, "")
    )
    monkeypatch.setattr(
        pymc_orchestrator, "load_pymc_model", lambda name, models_dir: object()
    )
    # Candidate admission now ends with a real MCMC fit-gate; stub it so the fake
    # stub candidates (not real PyMC models) are admitted without sampling.
    monkeypatch.setattr(pymc_orchestrator, "fit_model", lambda *a, **k: object())


def _patch_candidates(monkeypatch, captured_critique_paths):
    def fake_spawn(candidate_dir, **kwargs):
        (candidate_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text("People use H.\n", encoding="utf-8")
        # Record whether the candidate's context points at a critique file.
        context = (candidate_dir / "CONTEXT.md").read_text(encoding="utf-8")
        captured_critique_paths.append("critiques.md" in context)
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", fake_spawn)


def _patch_critique_agent(monkeypatch, spawn_log):
    """Stub the critique agent: write critiques.md, record the incumbent it saw."""

    def fake_spawn_critique(critique_dir, incumbent, **kwargs):
        spawn_log.append(incumbent)
        critique_dir.mkdir(parents=True, exist_ok=True)
        (critique_dir / "critiques.md").write_text(
            f"# Critique of {incumbent}\n\nIt under-predicts variance.\n",
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr(
        pymc_orchestrator, "_spawn_critique_agent", fake_spawn_critique
    )


def test_critique_default_significance_alpha():
    # The critique flags a discrepancy at raw two-sided p ≤ 0.05 (no multiple-
    # comparisons correction); --critique-alpha overrides it per run.
    import inspect

    from src.pipelines.inner_loop.pymc_orchestrator import (
        CRITIQUE_SIGNIFICANCE_ALPHA,
        run_pymc_inner_loop,
    )

    assert CRITIQUE_SIGNIFICANCE_ALPHA == 0.05
    default = inspect.signature(run_pymc_inner_loop).parameters[
        "critique_significance_alpha"
    ].default
    assert default == 0.05


def test_critique_module_defines_repo_root():
    # Regression: `_spawn_critique_agent` runs the critique agent with
    # `cwd=REPO_ROOT`. A missing module-level import made every critique skip with
    # "NameError: name 'REPO_ROOT' is not defined". Guard the symbol's presence.
    assert hasattr(pymc_orchestrator, "REPO_ROOT")


def test_critique_runs_before_each_candidate_round_and_feeds_candidates(
    tmp_path, monkeypatch
):
    _patch_scoring(
        monkeypatch,
        [
            _posterior("model_a", ["model_b"]),
            _posterior("iter0_candidate0", ["model_a", "model_b"]),
        ],
    )
    critique_incumbents: list = []
    candidate_saw_critique: list = []
    _patch_critique_agent(monkeypatch, critique_incumbents)
    _patch_candidates(monkeypatch, candidate_saw_critique)

    results_dir = tmp_path / "results"
    run_pymc_inner_loop(
        _make_responses(tmp_path),
        results_dir,
        seed_models_dir=_make_seed_models(tmp_path),
        max_iterations=1,
        candidate_count=1,
        enable_critique=True,
    )

    # The incumbent at the start of round 0 is model_a (the seed-set winner).
    assert critique_incumbents == ["model_a"]
    # The round's critiques.md was written and the candidate context pointed at it.
    assert (results_dir / "iter_0" / "critique" / "critiques.md").exists()
    assert candidate_saw_critique == [True]


def test_enable_critique_false_skips_the_critique(tmp_path, monkeypatch):
    _patch_scoring(
        monkeypatch,
        [
            _posterior("model_a", ["model_b"]),
            _posterior("iter0_candidate0", ["model_a", "model_b"]),
        ],
    )
    critique_incumbents: list = []
    candidate_saw_critique: list = []
    _patch_critique_agent(monkeypatch, critique_incumbents)
    _patch_candidates(monkeypatch, candidate_saw_critique)

    results_dir = tmp_path / "results"
    run_pymc_inner_loop(
        _make_responses(tmp_path),
        results_dir,
        seed_models_dir=_make_seed_models(tmp_path),
        max_iterations=1,
        candidate_count=1,
        enable_critique=False,
    )

    assert critique_incumbents == []
    assert not (results_dir / "iter_0" / "critique").exists()
    assert candidate_saw_critique == [False]
