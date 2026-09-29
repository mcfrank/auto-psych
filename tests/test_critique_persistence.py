"""The inner-loop orchestrator must always persist critique results.

The critique agent only proposes test statistics; the orchestrator runs the
posterior-predictive harness itself and writes ``ppc_results.json`` + a derived
``critiques.md`` so the critique results are recorded deterministically (not
left to whether the agent happened to run the harness).

The critique context is inlined into the agent's prompt (the file on disk is
for audit), and there is **no** pipeline-written fallback battery: with no
usable statistic the harness is not run at all — the caller retries the agent
once and then records the round as having no critique.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.critique.ppc as ppc
from src.pipelines.inner_loop import critique_round
from src.pipelines.inner_loop.critique_round import (
    _build_critique_prompt,
    _format_critiques_md,
    _persist_critique_results,
    _usable_test_statistics,
)
from src.runtime.config import REPO_ROOT

_RESULT = {
    "model": "bayesian_fair_coin",
    "n_test_statistics": 2,
    "n_replicates": 200,
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

_STAT_FILE = (
    "# name: alternation_gap\n"
    "# description: alternation proportion of A\n"
    "def test_statistic(df):\n    return 0.0\n"
)


def _write_stat(stats_dir: Path, name: str, source: str = _STAT_FILE) -> Path:
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_format_critiques_md_lists_significant_discrepancies():
    md = _format_critiques_md(_RESULT)
    assert "1 of 2" in md
    assert "Significant discrepancies" in md
    assert "alternation_gap" in md
    assert "max_run" not in md  # not significant -> not listed


# ─────────────────────────────────────────────
# The prompt carries the context inline
# ─────────────────────────────────────────────


def test_critique_prompt_inlines_the_context_and_names_the_output_dir(tmp_path: Path):
    """The critique agent's first action used to be reading CRITIQUE_CONTEXT.md;
    when that read was denied it wrote nothing and exited. The context is now a
    section of the prompt, exactly as the candidate agent's documents are."""
    crit = tmp_path / "critique"
    context = "# Critique context\n\n**Incumbent (best) model:** `seed_a`\n"
    prompt = _build_critique_prompt(crit, context)
    assert "## CRITIQUE_CONTEXT.md" in prompt
    assert context in prompt
    assert f"{crit}/test_stats" in prompt
    # Nothing in the prompt tells the agent to go and read the context file.
    assert "Read `CRITIQUE_CONTEXT.md`" not in prompt
    assert "second attempt" not in prompt


def test_critique_prompt_second_attempt_says_the_first_wrote_nothing(tmp_path: Path):
    crit = tmp_path / "critique"
    prompt = _build_critique_prompt(crit, "ctx", attempt=1)
    assert "second attempt" in prompt
    assert f"{crit}/test_stats" in prompt


def test_critique_prompt_refuses_an_empty_context(tmp_path: Path):
    with pytest.raises(ValueError):
        _build_critique_prompt(tmp_path / "critique", "")


def test_write_critique_context_returns_the_text_it_wrote(tmp_path: Path):
    responses = tmp_path / "responses.csv"
    responses.write_text("chose_left,n_a\n1,6\n0,6\n", encoding="utf-8")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "models_manifest.yaml").write_text(
        "models:\n  - name: seed_a\n    rationale: people use H\n", encoding="utf-8"
    )
    crit = tmp_path / "critique"
    text = critique_round._write_critique_context(
        crit, "seed_a", models_dir, responses, tmp_path / "cache",
        n_proposals=8, significance_alpha=0.05, n_replicates=200,
    )
    assert text == (crit / "CRITIQUE_CONTEXT.md").read_text(encoding="utf-8")
    assert "people use H" in text
    assert "chose_left,n_a" in text


def test_critique_prompt_names_the_critique_dir(tmp_path: Path, monkeypatch):
    """The critique agent runs from REPO_ROOT, so its prompt must name the critique
    dir explicitly — otherwise it cannot reliably locate its output directory."""
    import src.runtime.coding_agent as coding_agent

    monkeypatch.setattr(critique_round, "_seed_critique_fit_cache", lambda *a, **k: None)
    monkeypatch.setattr(critique_round, "_write_critique_context", lambda *a, **k: "ctx")
    monkeypatch.setattr(critique_round, "_persist_critique_results", lambda *a, **k: _RESULT)

    captured = {}

    def fake_run(
        prompt, *, cwd, log_path, allowed_dirs, timeout_secs, backend, usage_label,
        model=None, stock=False,
    ):
        captured["prompt"] = prompt
        captured["cwd"] = cwd
        captured["stock"] = stock
        _write_stat(log_path.parent / "test_stats", "alternation_gap")
        return True, ""

    monkeypatch.setattr(coding_agent, "run_coding_agent", fake_run)
    crit = tmp_path / "critique"
    critique_round._spawn_critique_agent(
        crit, "incumbent", models_dir=tmp_path, responses_path=tmp_path / "r.csv",
        cache_dir=None, fit_kwargs={}, n_proposals=8, significance_alpha=0.05,
        n_replicates=10, agent_timeout_sec=10, backend="opencode",
    )
    assert str(crit) in captured["prompt"]
    assert captured["cwd"] == REPO_ROOT
    # The critic is part of the experiment too: a stock agent.
    assert captured["stock"] is True


# ─────────────────────────────────────────────
# Usable statistics: the import gate, and no fallback
# ─────────────────────────────────────────────


def test_usable_test_statistics_removes_forbidden_imports_and_keeps_the_rest(tmp_path: Path):
    stats_dir = tmp_path / "test_stats"
    good = _write_stat(stats_dir, "good")
    _write_stat(
        stats_dir, "leaky",
        "# name: leaky\n# description: d\nfrom src.subjective_randomness import features\n"
        "def test_statistic(df):\n    return 0.0\n",
    )
    assert _usable_test_statistics(stats_dir) == [good]
    assert sorted(p.name for p in stats_dir.glob("*.py")) == ["good.py"]


def test_usable_test_statistics_is_empty_without_a_directory(tmp_path: Path):
    assert _usable_test_statistics(tmp_path / "missing") == []


def test_persist_raises_when_the_agent_wrote_no_statistics(tmp_path: Path, monkeypatch):
    """No fallback battery: with nothing to score, persisting is a caller bug
    and raises; the PPC harness is never run and nothing is written."""
    ppc_calls: list = []
    monkeypatch.setattr(ppc, "run_ppc_for_model", lambda *a, **k: ppc_calls.append(a))
    crit = tmp_path / "critique"
    (crit / "test_stats").mkdir(parents=True)
    with pytest.raises(ValueError, match="no usable test statistic"):
        _persist_critique_results(
            crit, "m", models_dir=tmp_path, responses_path=tmp_path / "r.csv",
            fit_cache_dir=tmp_path, fit_kwargs={}, n_replicates=10,
            significance_alpha=0.05,
        )
    assert ppc_calls == []
    assert not (crit / "ppc_results.json").exists()
    assert not (crit / "critiques.md").exists()


def test_persist_writes_results_and_critiques(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ppc, "run_ppc_for_model", lambda *a, **k: _RESULT)
    crit = tmp_path / "critique"
    _write_stat(crit / "test_stats", "alternation_gap")

    result = _persist_critique_results(
        crit, "bayesian_fair_coin", models_dir=tmp_path,
        responses_path=tmp_path / "r.csv", fit_cache_dir=tmp_path,
        fit_kwargs={}, n_replicates=200, significance_alpha=0.05,
    )

    assert result == _RESULT
    saved = json.loads((crit / "ppc_results.json").read_text())
    assert saved["n_significant"] == 1
    assert {r["name"] for r in saved["results"]} == {"alternation_gap", "max_run"}
    assert "alternation_gap" in (crit / "critiques.md").read_text()


# ─────────────────────────────────────────────
# Retry once, then no critique — and the recorded status
# ─────────────────────────────────────────────


def _spawn(tmp_path: Path, **overrides):
    kwargs = dict(
        models_dir=tmp_path, responses_path=tmp_path / "r.csv", cache_dir=None,
        fit_kwargs={}, n_proposals=8, significance_alpha=0.05, n_replicates=10,
        agent_timeout_sec=10, backend="opencode",
    )
    kwargs.update(overrides)
    return critique_round._spawn_critique_agent(tmp_path / "critique", "incumbent", **kwargs)


def _patch_agent(monkeypatch, on_run):
    import src.runtime.coding_agent as coding_agent

    monkeypatch.setattr(critique_round, "_seed_critique_fit_cache", lambda *a, **k: None)
    monkeypatch.setattr(critique_round, "_write_critique_context", lambda *a, **k: "ctx")
    calls: list = []

    def fake_run(
        prompt, *, cwd, log_path, allowed_dirs, timeout_secs, backend, usage_label,
        model=None, stock=False,
    ):
        calls.append({"prompt": prompt, "log_path": log_path})
        on_run(log_path.parent)
        return True, ""

    monkeypatch.setattr(coding_agent, "run_coding_agent", fake_run)
    return calls


def test_spawn_retries_once_when_no_statistics_then_reports_no_critique(
    tmp_path: Path, monkeypatch
):
    calls = _patch_agent(monkeypatch, on_run=lambda critique_dir: None)
    persisted: list = []
    monkeypatch.setattr(
        critique_round, "_persist_critique_results", lambda *a, **k: persisted.append(a)
    )

    status = _spawn(tmp_path)

    crit = tmp_path / "critique"
    assert [c["log_path"] for c in calls] == [
        crit / "agent.jsonl", crit / "agent.retry_1.jsonl",
    ]
    assert "second attempt" not in calls[0]["prompt"]
    assert "second attempt" in calls[1]["prompt"]
    assert persisted == []
    assert status == {
        "status": "no_critique",
        "incumbent": "incumbent",
        "attempts": 2,
        "reason": "the critique agent wrote no usable test statistic in 2 attempts",
    }


def test_spawn_retries_when_every_statistic_failed_the_import_gate(tmp_path: Path, monkeypatch):
    """A statistic importing forbidden code is not usable; a round of only
    those is retried like an empty one."""
    def write_leaky(critique_dir):
        _write_stat(
            critique_dir / "test_stats", "leaky",
            "# name: leaky\n# description: d\nimport pandas as pd\n"
            "def test_statistic(df):\n    return 0.0\n",
        )

    calls = _patch_agent(monkeypatch, on_run=write_leaky)
    monkeypatch.setattr(critique_round, "_persist_critique_results", lambda *a, **k: _RESULT)

    status = _spawn(tmp_path)

    assert len(calls) == 2
    assert status["status"] == "no_critique"


def test_spawn_does_not_retry_when_statistics_were_written(tmp_path: Path, monkeypatch):
    calls = _patch_agent(
        monkeypatch, on_run=lambda d: _write_stat(d / "test_stats", "alternation_gap")
    )
    monkeypatch.setattr(critique_round, "_persist_critique_results", lambda *a, **k: _RESULT)

    status = _spawn(tmp_path)

    assert len(calls) == 1
    assert status == {
        "status": "critiqued",
        "incumbent": "incumbent",
        "attempts": 1,
        "n_statistics": 2,
        "n_significant": 1,
        "n_significant_fdr": 0,
    }


def test_spawn_second_attempt_can_still_succeed(tmp_path: Path, monkeypatch):
    seen = {"n": 0}

    def flaky(critique_dir):
        seen["n"] += 1
        if seen["n"] == 2:
            _write_stat(critique_dir / "test_stats", "alternation_gap")

    calls = _patch_agent(monkeypatch, on_run=flaky)
    monkeypatch.setattr(critique_round, "_persist_critique_results", lambda *a, **k: _RESULT)

    status = _spawn(tmp_path)

    assert len(calls) == 2
    assert status["status"] == "critiqued"
    assert status["attempts"] == 2


def _run_round(tmp_path: Path):
    return critique_round._run_critique_round(
        tmp_path / "iter_0",
        responses_path=tmp_path / "responses.csv",
        models_dir=tmp_path / "models",
        posterior={"posteriors": {"seed": 1.0}, "elpd_loo": {"seed": -1.0}},
        comparison={},
        cache_dir=None,
        fit_kwargs={},
        n_proposals=1,
        significance_alpha=0.05,
        n_replicates=1,
        agent_timeout_sec=1,
        backend="opencode",
    )


def test_run_critique_round_returns_the_critiques_path_and_status(tmp_path: Path, monkeypatch):
    critiqued = {
        "status": "critiqued", "incumbent": "seed", "attempts": 1,
        "n_statistics": 2, "n_significant": 1, "n_significant_fdr": 0,
    }

    def fake_spawn(critique_dir, incumbent, **kwargs):
        critique_dir.mkdir(parents=True, exist_ok=True)
        (critique_dir / "critiques.md").write_text("# Critique of seed\n", encoding="utf-8")
        return critiqued

    monkeypatch.setattr(critique_round, "_spawn_critique_agent", fake_spawn)
    outcome = _run_round(tmp_path)
    assert outcome.critiques_md == tmp_path / "iter_0" / "critique" / "critiques.md"
    assert outcome.status == critiqued


def test_run_critique_round_without_statistics_feeds_no_critique(tmp_path: Path, monkeypatch):
    absent = {
        "status": "no_critique", "incumbent": "seed", "attempts": 2,
        "reason": "the critique agent wrote no usable test statistic in 2 attempts",
    }
    monkeypatch.setattr(critique_round, "_spawn_critique_agent", lambda *a, **k: absent)
    outcome = _run_round(tmp_path)
    assert outcome.critiques_md is None
    assert outcome.status == absent


def test_run_critique_round_records_a_harness_failure_as_no_critique(tmp_path: Path, monkeypatch):
    """A crash inside the critique (e.g. the PPC harness on a malformed
    statistic) must not kill a long inner-loop run — but it is recorded as
    "no critique" with the reason, not swallowed."""
    def boom(*a, **k):
        raise RuntimeError("statistic file has no test_statistic")

    monkeypatch.setattr(critique_round, "_spawn_critique_agent", boom)
    outcome = _run_round(tmp_path)
    assert outcome.critiques_md is None
    assert outcome.status == {
        "status": "no_critique",
        "incumbent": "seed",
        "reason": "RuntimeError: statistic file has no test_statistic",
    }


def test_run_critique_round_refuses_a_critiqued_status_without_critiques_md(
    tmp_path: Path, monkeypatch
):
    critiqued = {
        "status": "critiqued", "incumbent": "seed", "attempts": 1,
        "n_statistics": 1, "n_significant": 0, "n_significant_fdr": 0,
    }
    monkeypatch.setattr(critique_round, "_spawn_critique_agent", lambda *a, **k: critiqued)
    with pytest.raises(RuntimeError, match="critiques.md"):
        _run_round(tmp_path)
