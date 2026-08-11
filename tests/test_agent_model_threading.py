"""The coding-agent model id threads from configs down to run_coding_agent.

The holdout config's ``agent.model`` key was previously read by nothing: every
opencode agent silently ran the backend's default (Gemini). Running recovery
with a different hosted model (e.g. DeepSeek via Fireworks) requires the
chosen id to reach all three spawn sites — the outer implement agent and
the inner loop's candidate and critique agents.
"""

from __future__ import annotations

import src.runtime.coding_agent as coding_agent
from src.pipelines.inner_loop import pymc_orchestrator
from src.pipelines.outer_loop import orchestrator as orch

AGENT_MODEL = "fireworks-ai/accounts/fireworks/models/deepseek-v4-flash-0731"


def _capture_run_coding_agent(monkeypatch, target_module):
    captured = {}

    def fake_run_coding_agent(prompt, **kwargs):
        captured.update(kwargs)
        return True, "ok"

    monkeypatch.setattr(target_module, "run_coding_agent", fake_run_coding_agent)
    return captured


def test_spawn_cc_agent_threads_model(tmp_path, monkeypatch):
    captured = _capture_run_coding_agent(monkeypatch, orch)
    ok, _ = orch.spawn_cc_agent(agent_key="3_implement", exp_dir=tmp_path, model=AGENT_MODEL)
    assert ok
    assert captured["model"] == AGENT_MODEL


def test_candidate_agent_threads_model(tmp_path, monkeypatch):
    captured = _capture_run_coding_agent(monkeypatch, coding_agent)
    candidate_dir = tmp_path / "candidate_0"
    candidate_dir.mkdir()
    responses = tmp_path / "responses.csv"
    responses.write_text("a\n1\n", encoding="utf-8")
    docs = {
        "context": "c",
        "brief": "b",
        "existing_hypotheses": "h",
        "critiques": None,
    }
    pymc_orchestrator._spawn_candidate_agent(
        candidate_dir,
        docs,
        models_dir=tmp_path,
        responses_path=responses,
        agent_timeout_sec=10,
        backend="opencode",
        agent_model=AGENT_MODEL,
    )
    assert captured["model"] == AGENT_MODEL


def test_critique_agent_threads_model(tmp_path, monkeypatch):
    captured = _capture_run_coding_agent(monkeypatch, coding_agent)
    monkeypatch.setattr(
        pymc_orchestrator, "_seed_critique_fit_cache", lambda *a, **k: None
    )
    monkeypatch.setattr(
        pymc_orchestrator, "_write_critique_context", lambda *a, **k: None
    )
    monkeypatch.setattr(
        pymc_orchestrator, "_persist_critique_results", lambda *a, **k: None
    )
    responses = tmp_path / "responses.csv"
    responses.write_text("a\n1\n", encoding="utf-8")
    pymc_orchestrator._spawn_critique_agent(
        tmp_path / "critique",
        "incumbent_model",
        models_dir=tmp_path,
        responses_path=responses,
        cache_dir=None,
        fit_kwargs={},
        n_proposals=1,
        significance_alpha=0.05,
        n_replicates=10,
        agent_timeout_sec=10,
        backend="opencode",
        agent_model=AGENT_MODEL,
    )
    assert captured["model"] == AGENT_MODEL


def test_programmatic_wrapper_threads_agent_model(tmp_path, monkeypatch):
    exp_dir = tmp_path / "data" / "outer_loop" / "subjective_randomness" / "experiment1"
    (exp_dir / "cognitive_models").mkdir(parents=True)
    captured = {}

    def fake_inner_loop(responses_path, results_dir, **inner_kwargs):
        captured.update(inner_kwargs)
        return {"best_model": "stub_best"}

    monkeypatch.setattr(orch, "_pooled_response_rows", lambda e: [{"chose_left": "1"}])
    monkeypatch.setattr(orch, "_load_project_featurizer", lambda project_dir: None)
    monkeypatch.setattr(orch, "_write_feature_csv", lambda rows, fz, out: out)
    monkeypatch.setattr(orch, "_export_inner_loop_model", lambda e, l, *, best_model: e)
    monkeypatch.setattr(
        "src.pipelines.inner_loop.pymc_orchestrator.run_pymc_inner_loop",
        fake_inner_loop,
    )

    orch.run_inner_model_loop_programmatic(
        exp_dir,
        max_iterations=1,
        candidate_count=1,
        agent_model=AGENT_MODEL,
    )

    assert captured["agent_model"] == AGENT_MODEL
