"""Tests for run-scoped LLM token-usage tracking.

The pipeline spends tokens in two places: coding-agent subprocesses (Claude
Code / opencode CLIs, spawned by ``run_coding_agent``) and hosted-API LLM calls
(``invoke_llm``). Every spend must land in the process-wide usage log so a run
can report its total token consumption.
"""

from __future__ import annotations

import json
import os
import stat
import threading

import pytest

from src.runtime import token_usage


@pytest.fixture(autouse=True)
def _fresh_usage_log():
    """Each test starts with an empty in-memory log and no sink."""
    token_usage.reset_usage_log()
    yield
    token_usage.reset_usage_log()


def _write_fake_cli(bin_dir, name: str, stdout_lines: list[str]) -> None:
    """Install an executable that prints canned lines and exits 0."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / name
    body = "\n".join(f"echo '{line}'" for line in stdout_lines)
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


# ─────────────────────────────────────────────
# Integration: run_coding_agent records usage
# ─────────────────────────────────────────────


OPENCODE_EVENTS = [
    json.dumps(
        {
            "type": "step_start",
            "part": {"type": "step-start"},
        }
    ),
    json.dumps(
        {
            "type": "text",
            "part": {"type": "text", "text": "hello from the agent"},
        }
    ),
    json.dumps(
        {
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "tokens": {
                    "total": 110,
                    "input": 100,
                    "output": 4,
                    "reasoning": 6,
                    "cache": {"write": 0, "read": 0},
                },
                "cost": 0.01,
            },
        }
    ),
    json.dumps(
        {
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "tokens": {
                    "total": 260,
                    "input": 200,
                    "output": 10,
                    "reasoning": 20,
                    "cache": {"write": 25, "read": 5},
                },
                "cost": 0.02,
            },
        }
    ),
]


def test_run_coding_agent_opencode_records_usage(tmp_path, monkeypatch):
    """An opencode run sums step_finish tokens into one labelled usage record."""
    from src.runtime.coding_agent import run_coding_agent

    _write_fake_cli(tmp_path / "bin", "opencode", OPENCODE_EVENTS)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    marker = token_usage.start_usage_log(tmp_path / "usage.jsonl")

    success, result = run_coding_agent(
        "do the thing",
        cwd=tmp_path,
        log_path=tmp_path / "log.jsonl",
        backend="opencode",
        usage_label="inner:candidate",
        on_summary=None,
    )

    assert success
    assert result == "hello from the agent"

    records = token_usage.records_since(marker)
    assert len(records) == 1
    rec = records[0]
    assert rec.source == "inner:candidate"
    assert rec.backend == "opencode"
    assert rec.input_tokens == 300
    assert rec.output_tokens == 14
    assert rec.reasoning_tokens == 26
    assert rec.cache_write_tokens == 25
    assert rec.cache_read_tokens == 5
    assert rec.cost_usd == pytest.approx(0.03)
    assert not rec.usage_missing

    # The record is also persisted as one JSONL line.
    lines = (tmp_path / "usage.jsonl").read_text().splitlines()
    assert len(lines) == 1
    on_disk = json.loads(lines[0])
    assert on_disk["source"] == "inner:candidate"
    assert on_disk["input_tokens"] == 300


CLAUDE_EVENTS = [
    json.dumps(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "working"}]},
        }
    ),
    json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "result": "done",
            "num_turns": 2,
            "total_cost_usd": 0.05,
            "usage": {
                "input_tokens": 40,
                "output_tokens": 15,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 2000,
            },
        }
    ),
]


def test_run_coding_agent_claude_records_usage(tmp_path, monkeypatch):
    """A claude run records the terminal result event's cumulative usage."""
    from src.runtime.coding_agent import run_coding_agent

    _write_fake_cli(tmp_path / "bin", "claude", CLAUDE_EVENTS)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    marker = token_usage.start_usage_log(tmp_path / "usage.jsonl")

    success, result = run_coding_agent(
        "do the thing",
        cwd=tmp_path,
        log_path=tmp_path / "log.jsonl",
        backend="claude",
        usage_label="outer:3_implement",
        on_summary=None,
    )

    assert success
    assert result == "done"

    records = token_usage.records_since(marker)
    assert len(records) == 1
    rec = records[0]
    assert rec.source == "outer:3_implement"
    assert rec.backend == "claude"
    assert rec.input_tokens == 40
    assert rec.output_tokens == 15
    assert rec.cache_write_tokens == 1000
    assert rec.cache_read_tokens == 2000
    assert rec.reasoning_tokens == 0
    assert rec.cost_usd == pytest.approx(0.05)
    assert not rec.usage_missing


def test_run_coding_agent_without_usage_events_flags_missing(tmp_path, monkeypatch):
    """A run whose stream carried no usage info still leaves a (flagged) record."""
    from src.runtime.coding_agent import run_coding_agent

    _write_fake_cli(tmp_path / "bin", "opencode", ["plain text, no JSON events"])
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    marker = token_usage.start_usage_log(tmp_path / "usage.jsonl")

    success, _result = run_coding_agent(
        "p",
        cwd=tmp_path,
        log_path=tmp_path / "log.jsonl",
        backend="opencode",
        usage_label="inner:candidate",
        on_summary=None,
    )

    assert success
    records = token_usage.records_since(marker)
    assert len(records) == 1
    assert records[0].usage_missing
    assert records[0].total_tokens == 0


# ─────────────────────────────────────────────
# Integration: opencode "database is locked" retry
# ─────────────────────────────────────────────
#
# opencode 1.17+ keeps its sessions in one shared sqlite database; when the
# inner loop spawns several candidate agents at once, the late arrivals can
# die instantly with "database is locked" during session creation. That
# failure is transient, so run_coding_agent must retry it (with backoff)
# rather than silently losing 2 of 3 candidates per round.


def _write_flaky_locked_cli(bin_dir, name: str, n_lock_failures: int, success_lines):
    """A fake CLI that fails with "database is locked" its first N calls.

    Invocation count is kept in a state file next to the script so retries
    (separate processes) see it. Returns the state-file path.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    count_file = bin_dir / f"{name}.count"
    script = bin_dir / name
    success_body = "\n".join(f"echo '{line}'" for line in success_lines)
    script.write_text(
        f"""#!/bin/sh
n=$(cat "{count_file}" 2>/dev/null || echo 0)
n=$((n+1)); echo $n > "{count_file}"
if [ $n -le {n_lock_failures} ]; then
  echo "Error: Unexpected error"
  echo ""
  echo "database is locked"
  exit 1
fi
{success_body}
exit 0
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return count_file


@pytest.fixture()
def _no_retry_backoff(monkeypatch):
    import src.runtime.coding_agent as coding_agent

    monkeypatch.setattr(coding_agent, "OPENCODE_LOCK_BACKOFF_SECS", 0.0)


def test_opencode_locked_db_is_retried(tmp_path, monkeypatch, _no_retry_backoff):
    from src.runtime.coding_agent import run_coding_agent

    count_file = _write_flaky_locked_cli(
        tmp_path / "bin", "opencode", n_lock_failures=2, success_lines=OPENCODE_EVENTS
    )
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    marker = token_usage.start_usage_log(tmp_path / "usage.jsonl")

    success, result = run_coding_agent(
        "p",
        cwd=tmp_path,
        log_path=tmp_path / "log.jsonl",
        backend="opencode",
        usage_label="inner:candidate",
        on_summary=None,
    )

    assert success
    assert result == "hello from the agent"
    assert count_file.read_text().strip() == "3"
    # One logical call -> one usage record, from the attempt that ran.
    records = token_usage.records_since(marker)
    assert len(records) == 1
    assert not records[0].usage_missing
    assert records[0].input_tokens == 300


def test_opencode_locked_db_retries_exhaust(tmp_path, monkeypatch, _no_retry_backoff):
    import src.runtime.coding_agent as coding_agent
    from src.runtime.coding_agent import run_coding_agent

    count_file = _write_flaky_locked_cli(
        tmp_path / "bin", "opencode", n_lock_failures=99, success_lines=[]
    )
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    marker = token_usage.start_usage_log(tmp_path / "usage.jsonl")

    success, result = run_coding_agent(
        "p",
        cwd=tmp_path,
        log_path=tmp_path / "log.jsonl",
        backend="opencode",
        usage_label="inner:candidate",
        on_summary=None,
    )

    assert not success
    assert "database is locked" in result
    assert (
        count_file.read_text().strip()
        == str(1 + coding_agent.OPENCODE_LOCK_RETRIES)
    )
    records = token_usage.records_since(marker)
    assert len(records) == 1
    assert records[0].usage_missing


def test_opencode_other_failure_is_not_retried(tmp_path, monkeypatch, _no_retry_backoff):
    from src.runtime.coding_agent import run_coding_agent

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    count_file = bin_dir / "opencode.count"
    script = bin_dir / "opencode"
    script.write_text(
        f"""#!/bin/sh
n=$(cat "{count_file}" 2>/dev/null || echo 0)
n=$((n+1)); echo $n > "{count_file}"
echo "Error: model not found"
exit 1
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    success, _result = run_coding_agent(
        "p",
        cwd=tmp_path,
        log_path=tmp_path / "log.jsonl",
        backend="opencode",
        usage_label="inner:candidate",
        on_summary=None,
    )

    assert not success
    assert count_file.read_text().strip() == "1"


# ─────────────────────────────────────────────
# Unit: the usage log itself
# ─────────────────────────────────────────────


def test_record_usage_accumulates_and_summarizes():
    marker = token_usage.records_marker()
    token_usage.record_usage(
        source="outer:3_implement",
        backend="claude",
        model="m",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.01,
    )
    token_usage.record_usage(
        source="participant",
        backend="langchain",
        model="gemini",
        input_tokens=100,
        output_tokens=50,
        reasoning_tokens=25,
        cache_read_tokens=7,
    )
    summary = token_usage.summarize(token_usage.records_since(marker))
    assert summary["n_calls"] == 2
    assert summary["input_tokens"] == 110
    assert summary["output_tokens"] == 55
    assert summary["reasoning_tokens"] == 25
    assert summary["cache_read_tokens"] == 7
    assert summary["total_tokens"] == 110 + 55 + 25 + 7
    assert summary["cost_usd"] == pytest.approx(0.01)
    assert summary["n_calls_missing_usage"] == 0
    assert set(summary["by_source"]) == {"outer:3_implement", "participant"}
    assert summary["by_source"]["participant"]["total_tokens"] == 182


def test_summarize_counts_missing_usage():
    token_usage.record_usage(
        source="s", backend="opencode", model="m", usage_missing=True
    )
    summary = token_usage.summarize(token_usage.records_since(0))
    assert summary["n_calls"] == 1
    assert summary["n_calls_missing_usage"] == 1
    assert summary["total_tokens"] == 0
    assert summary["cost_usd"] is None


def test_start_usage_log_appends_across_starts(tmp_path):
    """Re-pointing the sink at the same file appends rather than truncating."""
    sink = tmp_path / "usage.jsonl"
    token_usage.start_usage_log(sink)
    token_usage.record_usage(
        source="a", backend="opencode", model="m", input_tokens=1
    )
    token_usage.start_usage_log(sink)
    token_usage.record_usage(
        source="b", backend="opencode", model="m", input_tokens=2
    )
    assert len(sink.read_text().splitlines()) == 2


def test_record_usage_is_thread_safe(tmp_path):
    token_usage.start_usage_log(tmp_path / "usage.jsonl")

    def _spam():
        for _ in range(50):
            token_usage.record_usage(
                source="t", backend="opencode", model="m", input_tokens=1
            )

    threads = [threading.Thread(target=_spam) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(token_usage.records_since(0)) == 400
    assert len((tmp_path / "usage.jsonl").read_text().splitlines()) == 400


# ─────────────────────────────────────────────
# Unit: invoke_llm records hosted-API usage
# ─────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, usage_metadata):
        self.content = "ACTION: click A"
        if usage_metadata is not None:
            self.usage_metadata = usage_metadata


class _FakeLLM:
    model = "models/gemini-test"

    def __init__(self, usage_metadata):
        self._usage_metadata = usage_metadata

    def invoke(self, messages, **kwargs):
        return _FakeResponse(self._usage_metadata)


def test_invoke_llm_records_usage_metadata():
    from src.pipelines.outer_loop.llm import invoke_llm

    marker = token_usage.records_marker()
    llm = _FakeLLM(
        {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "input_token_details": {"cache_read": 20},
            "output_token_details": {"reasoning": 10},
        }
    )
    reply = invoke_llm(system="s", user="u", llm=llm, source="participant")

    assert reply == "ACTION: click A"
    records = token_usage.records_since(marker)
    assert len(records) == 1
    rec = records[0]
    assert rec.source == "participant"
    assert rec.model == "models/gemini-test"
    # Components are disjoint: cache reads out of input, reasoning out of output.
    assert rec.input_tokens == 100
    assert rec.cache_read_tokens == 20
    assert rec.output_tokens == 20
    assert rec.reasoning_tokens == 10
    assert rec.total_tokens == 150
    assert not rec.usage_missing


def test_invoke_llm_flags_missing_usage_metadata():
    from src.pipelines.outer_loop.llm import invoke_llm

    marker = token_usage.records_marker()
    invoke_llm(system="s", user="u", llm=_FakeLLM(None))

    records = token_usage.records_since(marker)
    assert len(records) == 1
    assert records[0].usage_missing
    assert records[0].total_tokens == 0


# ─────────────────────────────────────────────
# Unit: pipeline call sites label their usage
# ─────────────────────────────────────────────


def _capture_run_coding_agent(monkeypatch, target_module):
    """Stub run_coding_agent on ``target_module``; return the captured kwargs."""
    captured = {}

    def fake_run_coding_agent(prompt, **kwargs):
        captured.update(kwargs)
        return True, "ok"

    monkeypatch.setattr(target_module, "run_coding_agent", fake_run_coding_agent)
    return captured


def test_spawn_cc_agent_labels_usage_by_stage(tmp_path, monkeypatch):
    from src.pipelines.outer_loop import orchestrator as orch

    captured = _capture_run_coding_agent(monkeypatch, orch)
    ok, _ = orch.spawn_cc_agent(agent_key="3_implement", exp_dir=tmp_path)
    assert ok
    assert captured["usage_label"] == "outer:3_implement"


def test_candidate_agent_labels_usage(tmp_path, monkeypatch):
    import src.runtime.coding_agent as coding_agent
    from src.pipelines.inner_loop import pymc_orchestrator

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
    )
    assert captured["usage_label"] == "inner:candidate"


def test_critique_agent_labels_usage(tmp_path, monkeypatch):
    import src.runtime.coding_agent as coding_agent
    from src.pipelines.inner_loop import pymc_orchestrator

    captured = _capture_run_coding_agent(monkeypatch, coding_agent)
    # The critique spawn seeds a PyMC fit cache and runs the PPC harness around
    # the agent call; both are irrelevant to the label under test.
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
    )
    assert captured["usage_label"] == "inner:critique"


def test_participant_model_labels_usage(monkeypatch):
    from src.pipelines.outer_loop import llm as llm_mod
    from src.pipelines.outer_loop.participants import ClosedParticipantModel

    fake = _FakeLLM({"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
    monkeypatch.setattr(llm_mod, "get_llm", lambda **kwargs: fake)

    marker = token_usage.records_marker()
    participant = ClosedParticipantModel()
    participant.answer("system", "user")

    records = token_usage.records_since(marker)
    assert len(records) == 1
    assert records[0].source == "participant"


# ─────────────────────────────────────────────
# Run-level wiring: logs and summaries on disk
# ─────────────────────────────────────────────


def test_write_usage_report_persists_and_prints(tmp_path, capsys):
    marker = token_usage.records_marker()
    token_usage.record_usage(
        source="outer:3_implement",
        backend="opencode",
        model="m",
        input_tokens=5,
        output_tokens=2,
    )
    summary = token_usage.write_usage_report(tmp_path, marker, heading="experiment 1")

    on_disk = json.loads((tmp_path / "token_usage_summary.json").read_text())
    assert on_disk == summary
    assert on_disk["total_tokens"] == 7
    assert on_disk["by_source"]["outer:3_implement"]["n_calls"] == 1
    out = capsys.readouterr().out
    assert "experiment 1" in out
    assert "7" in out


def test_run_experiment_persists_usage_log(tmp_path, monkeypatch):
    import src.pipelines.outer_loop.run as run_mod

    exp_dir = tmp_path / "experiment1"
    monkeypatch.setattr(run_mod, "experiment_dir", lambda p, n: exp_dir)
    monkeypatch.setattr(
        run_mod, "ensure_experiment_dirs", lambda d: d.mkdir(parents=True, exist_ok=True)
    )
    monkeypatch.setattr(run_mod, "init_registry", lambda d: None)
    monkeypatch.setattr(
        run_mod, "seed_experiment_models_from_project", lambda d, p: False
    )

    def fake_run_agent(**kwargs):
        token_usage.record_usage(
            source="outer:3_implement", backend="opencode", model="m", input_tokens=3
        )

    monkeypatch.setattr(run_mod, "_run_agent", fake_run_agent)

    run_mod._run_experiment(
        project_id="p",
        exp_num=1,
        mode="simulated_participants",
        n_participants=1,
        validate=False,
        agent_filter="3_implement",
    )

    assert (exp_dir / "token_usage.jsonl").exists()
    summary = json.loads((exp_dir / "token_usage_summary.json").read_text())
    assert summary["total_tokens"] == 3


def test_inner_cli_persists_usage_log(tmp_path, monkeypatch):
    import src.pipelines.inner_loop.run as inner_run

    responses = tmp_path / "responses.csv"
    responses.write_text("a\n1\n", encoding="utf-8")
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "models_manifest.yaml").write_text("models: []\n", encoding="utf-8")
    results = tmp_path / "results"

    def fake_inner_loop(*args, **kwargs):
        token_usage.record_usage(
            source="inner:candidate", backend="opencode", model="m", input_tokens=9
        )
        return {
            "best_model": "m",
            "posteriors": {"m": 1.0},
            "elpd_loo": {"m": -1.0},
        }

    monkeypatch.setattr(inner_run, "run_pymc_inner_loop", fake_inner_loop)

    inner_run.main(
        inner_run.Args(responses=responses, seed_models=seeds, results=results)
    )

    assert (results / "token_usage.jsonl").exists()
    summary = json.loads((results / "token_usage_summary.json").read_text())
    assert summary["total_tokens"] == 9
