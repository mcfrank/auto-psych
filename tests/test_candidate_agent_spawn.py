"""The candidate agent must be spawned so opencode loads its directory grants.

opencode sandboxes file access to its working directory and only reaches paths
outside it through the ``external_directory`` grants in the run copy's
``opencode.json`` — which it discovers by walking up from its ``cwd``. The run
harness patches those grants into the worktree's ``opencode.json`` (``REPO_ROOT``
at run time), so an agent only sees them when it runs *from* the worktree.

The candidate agent previously ran with ``cwd=candidate_dir`` on ``$SCRATCH``,
outside the worktree, so opencode found no ``opencode.json``, loaded no grants,
and auto-rejected every external path the agent's CONTEXT.md pointed it at
(``critiques.md``, the responses CSV, the model set). The result: no
``candidate.py`` was ever written and the model set never grew.

It must instead run from ``REPO_ROOT`` like the critique agent, and (for the
Claude backend, which honours ``allowed_dirs`` via ``--add-dir``) be granted the
responses and model-set directories.
"""

from __future__ import annotations

import src.runtime.coding_agent as coding_agent
from src.pipelines.inner_loop import pymc_orchestrator
from src.runtime.config import REPO_ROOT


def _spawn(tmp_path, monkeypatch):
    """Spawn a candidate agent with run_coding_agent stubbed; return its kwargs."""
    captured = {}

    def fake_run_coding_agent(
        prompt, *, cwd, log_path, allowed_dirs, timeout_secs, backend
    ):
        captured["prompt"] = prompt
        captured["cwd"] = cwd
        captured["allowed_dirs"] = list(allowed_dirs)
        captured["backend"] = backend
        return True, ""

    monkeypatch.setattr(coding_agent, "run_coding_agent", fake_run_coding_agent)

    candidate_dir = tmp_path / "model_loop" / "iter_0" / "candidate_0"
    candidate_dir.mkdir(parents=True)
    models_dir = tmp_path / "model_loop" / "models"
    models_dir.mkdir(parents=True)
    responses_path = tmp_path / "model_loop" / "responses.csv"
    responses_path.write_text("a,b\n1,2\n", encoding="utf-8")

    docs = {
        "context": "ctx",
        "brief": "brief",
        "existing_hypotheses": "hyps",
        "critiques": None,
    }
    pymc_orchestrator._spawn_candidate_agent(
        candidate_dir,
        docs,
        models_dir=models_dir,
        responses_path=responses_path,
        agent_timeout_sec=10,
        backend="opencode",
    )
    return captured, candidate_dir, models_dir, responses_path


def test_candidate_agent_runs_from_repo_root(tmp_path, monkeypatch):
    captured, *_ = _spawn(tmp_path, monkeypatch)
    # Running from REPO_ROOT is what lets opencode discover the worktree's
    # opencode.json external_directory grants; candidate_dir on $SCRATCH does not.
    assert captured["cwd"] == REPO_ROOT


def test_candidate_agent_is_granted_data_and_model_dirs(tmp_path, monkeypatch):
    captured, candidate_dir, models_dir, responses_path = _spawn(tmp_path, monkeypatch)
    allowed = captured["allowed_dirs"]
    # The candidate's own workspace, the responses CSV, and the model set must all
    # be reachable (mirrors the critique agent's allowed_dirs for the Claude
    # backend, which honours these via --add-dir).
    assert candidate_dir in allowed
    assert responses_path.parent in allowed
    assert models_dir in allowed
