"""Inner-loop integration tests for the CriticAL critique round.

Before each candidate-generation round the inner loop critiques the current
incumbent (best) model: it spawns a critique agent that posterior-predictively
checks the incumbent and writes ``critiques.md``, then passes that critique to
the candidate agents so they target the model's actual failure modes. MCMC and
both agent spawns are stubbed — these tests cover only the orchestration:

* a critique round runs before candidates each iteration,
* the candidate context points at the round's ``critiques.md``,
* ``enable_critique=False`` skips the critique entirely,
* a critique agent that writes no usable test statistic is re-spawned once;
  if the retry also writes none, the round proceeds with **no** critique (no
  pipeline-written fallback battery — that hid a dead critique subsystem
  through a full sweep) and ``history.json`` records the absence,
* every round's critique status (statistics proposed / significant, or "no
  critique", or disabled) is recorded in ``history.json``.
"""

from __future__ import annotations

import json

import src.critique.ppc as ppc
import src.pipelines.inner_loop.critique_round as critique_round
import src.pipelines.inner_loop.model_zoo as model_zoo
import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
import src.pipelines.inner_loop.scoring as scoring
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from tests.inner_loop_fixtures import canned_posterior, write_responses, write_seed_models


def _patch_scoring(monkeypatch, posteriors_per_call):
    calls = {"n": 0}

    def fake_model_posterior(responses_path, models_dir, **kwargs):
        result = posteriors_per_call[calls["n"]]
        calls["n"] += 1
        return result

    monkeypatch.setattr(scoring, "model_posterior", fake_model_posterior)
    monkeypatch.setattr(scoring, "compare_table", lambda *a, **k: {})
    # _prune_losers looks up compare_table in model_zoo's namespace:
    monkeypatch.setattr(model_zoo, "compare_table", lambda *a, **k: {})
    # Functions looked up in model_zoo's namespace:
    monkeypatch.setattr(
        model_zoo, "model_logp_is_finite", lambda *a, **k: (True, "")
    )
    monkeypatch.setattr(
        model_zoo, "load_pymc_model", lambda name, models_dir: object()
    )
    # Candidate admission now ends with a real MCMC fit-gate; stub it so the fake
    # stub candidates (not real PyMC models) are admitted without sampling.
    monkeypatch.setattr(model_zoo, "fit_model", lambda *a, **k: object())
    # The experiment-start screen samples the whole set in one batch; no MCMC here.
    monkeypatch.setattr(model_zoo, "fit_models_to_cache", lambda names, *a, **k: {})
    # Admission also gates on a finite ELPD-LOO; stub it finite for stub candidates.
    monkeypatch.setattr(model_zoo, "log_likelihood", lambda *a, **k: -100.0)
    # Novelty gate is covered by test_novelty_gate.py; neutralize it here.
    monkeypatch.setattr(
        model_zoo, "_min_prediction_rmse",
        lambda *a, **k: (None, float("inf")),
    )


def _patch_candidates(monkeypatch, captured_critique_paths):
    def fake_spawn(candidate_dir, docs, **kwargs):
        (candidate_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text("People use H.\n", encoding="utf-8")
        # Record whether the critique was injected into the candidate's docs
        # (it is inlined into the prompt, not merely pointed at).
        captured_critique_paths.append(bool(docs.get("critiques")))
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
        return {
            "status": "critiqued", "incumbent": incumbent, "attempts": 1,
            "n_statistics": 1, "n_significant": 1, "n_significant_fdr": 0,
        }

    monkeypatch.setattr(
        critique_round, "_spawn_critique_agent", fake_spawn_critique
    )


def test_critique_default_significance_alpha():
    # The critique flags a discrepancy at raw two-sided p ≤ 0.05 (no multiple-
    # comparisons correction); --critique-alpha overrides it per run.
    import inspect

    from src.pipelines.inner_loop.critique_round import CRITIQUE_SIGNIFICANCE_ALPHA
    from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop

    assert CRITIQUE_SIGNIFICANCE_ALPHA == 0.05
    default = inspect.signature(run_pymc_inner_loop).parameters[
        "critique_significance_alpha"
    ].default
    assert default == 0.05


def test_critique_module_defines_repo_root():
    # Regression: `_spawn_critique_agent` runs the critique agent with
    # `cwd=REPO_ROOT`. A missing module-level import made every critique skip with
    # "NameError: name 'REPO_ROOT' is not defined". Guard the symbol's presence.
    assert hasattr(critique_round, "REPO_ROOT")


def test_critique_runs_before_each_candidate_round_and_feeds_candidates(
    tmp_path, monkeypatch
):
    _patch_scoring(
        monkeypatch,
        [
            canned_posterior("model_a", ["model_b"]),
            canned_posterior("iter0_candidate0", ["model_a", "model_b"]),
        ],
    )
    critique_incumbents: list = []
    candidate_saw_critique: list = []
    _patch_critique_agent(monkeypatch, critique_incumbents)
    _patch_candidates(monkeypatch, candidate_saw_critique)

    results_dir = tmp_path / "results"
    run_pymc_inner_loop(
        write_responses(tmp_path),
        results_dir,
        seed_models_dir=write_seed_models(tmp_path),
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
            canned_posterior("model_a", ["model_b"]),
            canned_posterior("iter0_candidate0", ["model_a", "model_b"]),
        ],
    )
    critique_incumbents: list = []
    candidate_saw_critique: list = []
    _patch_critique_agent(monkeypatch, critique_incumbents)
    _patch_candidates(monkeypatch, candidate_saw_critique)

    results_dir = tmp_path / "results"
    run_pymc_inner_loop(
        write_responses(tmp_path),
        results_dir,
        seed_models_dir=write_seed_models(tmp_path),
        max_iterations=1,
        candidate_count=1,
        enable_critique=False,
    )

    assert critique_incumbents == []
    assert not (results_dir / "iter_0" / "critique").exists()
    assert candidate_saw_critique == [False]
    history = json.loads((results_dir / "history.json").read_text(encoding="utf-8"))
    assert history[1]["critique"] == {"status": "disabled"}


# ─────────────────────────────────────────────
# Retry-then-skip, and the recorded per-round status
# ─────────────────────────────────────────────


def _patch_critique_agent_process(monkeypatch, on_run):
    """Stub the coding-agent subprocess under the real critique round.

    ``on_run(critique_dir)`` plays the agent: it may write test statistics
    into ``critique_dir/test_stats``. Returns the list of calls (prompt + log
    path) so a test can count attempts and inspect the prompt. The fit-cache
    seeding is stubbed (it would run MCMC).
    """
    import src.runtime.coding_agent as coding_agent

    monkeypatch.setattr(critique_round, "_seed_critique_fit_cache", lambda *a, **k: None)
    calls: list = []

    def fake_run(
        prompt, *, cwd, log_path, allowed_dirs, timeout_secs, backend, usage_label,
        model=None,
    ):
        calls.append({"prompt": prompt, "log_path": log_path})
        on_run(log_path.parent)
        return True, ""

    monkeypatch.setattr(coding_agent, "run_coding_agent", fake_run)
    return calls


_PPC_RESULT = {
    "model": "model_a",
    "n_test_statistics": 2,
    "n_replicates": 10,
    "significance_alpha": 0.05,
    "n_significant": 1,
    "n_significant_fdr": 0,
    "results": [
        {"name": "alternation_gap", "description": "alternation proportion of A",
         "t_observed": 0.55, "null_mean": 0.50, "null_std": 0.02, "z_score": 2.5,
         "p_value": 0.012, "p_value_fdr": 0.024, "significant": True,
         "significant_fdr": False, "error": None},
        {"name": "max_run", "description": "max run length",
         "t_observed": 2.1, "null_mean": 2.0, "null_std": 0.3, "z_score": 0.5,
         "p_value": 0.6, "p_value_fdr": 0.6, "significant": False,
         "significant_fdr": False, "error": None},
    ],
}


def _write_one_statistic(critique_dir):
    stats_dir = critique_dir / "test_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "alternation_gap.py").write_text(
        "# name: alternation_gap\n# description: alternation proportion of A\n"
        "def test_statistic(df):\n    return 0.0\n",
        encoding="utf-8",
    )


def test_critique_agent_writing_no_statistics_is_retried_once_then_round_has_no_critique(
    tmp_path, monkeypatch
):
    """Zero statistics ⇒ one retry; still zero ⇒ no critiques.md, the PPC
    harness never runs (there is no fallback battery any more), the candidates
    run without a critique, and history.json says so."""
    _patch_scoring(
        monkeypatch,
        [
            canned_posterior("model_a", ["model_b"]),
            canned_posterior("iter0_candidate0", ["model_a", "model_b"]),
        ],
    )
    candidate_saw_critique: list = []
    _patch_candidates(monkeypatch, candidate_saw_critique)
    calls = _patch_critique_agent_process(monkeypatch, on_run=lambda critique_dir: None)
    ppc_calls: list = []
    monkeypatch.setattr(ppc, "run_ppc_for_model", lambda *a, **k: ppc_calls.append(a))

    results_dir = tmp_path / "results"
    run_pymc_inner_loop(
        write_responses(tmp_path),
        results_dir,
        seed_models_dir=write_seed_models(tmp_path),
        max_iterations=1,
        candidate_count=1,
        enable_critique=True,
    )

    critique_dir = results_dir / "iter_0" / "critique"
    assert [c["log_path"] for c in calls] == [
        critique_dir / "agent.jsonl",
        critique_dir / "agent.retry_1.jsonl",
    ]
    assert "second attempt" in calls[1]["prompt"]
    assert "second attempt" not in calls[0]["prompt"]
    assert ppc_calls == []
    assert not (critique_dir / "critiques.md").exists()
    assert not (critique_dir / "ppc_results.json").exists()
    assert candidate_saw_critique == [False]

    history = json.loads((results_dir / "history.json").read_text(encoding="utf-8"))
    assert "critique" not in history[0]  # the seed scoring step precedes any round
    status = history[1]["critique"]
    assert status["status"] == "no_critique"
    assert status["incumbent"] == "model_a"
    assert status["attempts"] == 2
    assert "no usable test statistic" in status["reason"]


def test_history_records_the_critique_statistics_per_round(tmp_path, monkeypatch):
    """A critique agent that writes statistics is not retried; the PPC results
    are summarised into the round's history entry, and the agent's prompt
    carried the critique context inline (it never had to read the file)."""
    _patch_scoring(
        monkeypatch,
        [
            canned_posterior("model_a", ["model_b"]),
            canned_posterior("iter0_candidate0", ["model_a", "model_b"]),
        ],
    )
    candidate_saw_critique: list = []
    _patch_candidates(monkeypatch, candidate_saw_critique)
    calls = _patch_critique_agent_process(monkeypatch, on_run=_write_one_statistic)
    monkeypatch.setattr(ppc, "run_ppc_for_model", lambda *a, **k: dict(_PPC_RESULT))

    results_dir = tmp_path / "results"
    run_pymc_inner_loop(
        write_responses(tmp_path),
        results_dir,
        seed_models_dir=write_seed_models(tmp_path),
        max_iterations=1,
        candidate_count=1,
        enable_critique=True,
    )

    critique_dir = results_dir / "iter_0" / "critique"
    assert len(calls) == 1
    context_on_disk = (critique_dir / "CRITIQUE_CONTEXT.md").read_text(encoding="utf-8")
    assert context_on_disk in calls[0]["prompt"]
    assert (critique_dir / "critiques.md").exists()
    assert candidate_saw_critique == [True]

    history = json.loads((results_dir / "history.json").read_text(encoding="utf-8"))
    assert history[1]["critique"] == {
        "status": "critiqued",
        "incumbent": "model_a",
        "attempts": 1,
        "n_statistics": 2,
        "n_significant": 1,
        "n_significant_fdr": 0,
    }
