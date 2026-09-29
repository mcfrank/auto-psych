"""An agent account's usage or rate limit is waited out, never a failed candidate.

On 2026-09-28 the Claude subscription of an Opus sweep ran out: every agent call
ended at once with "You've hit your session limit · resets 2:20pm
(America/Los_Angeles)" (a ``result`` event of subtype "success"), and the loop
read each as an agent that wrote no model. Slots were retried, rounds
abandoned, and a cell raised ``AllCandidatesNoFileError`` (5 of 5 rounds
abandoned); a completed cell silently lost a round. One agent had worked for
ten turns before the limit cut it off, so a half-written candidate could be
left behind.

The stub CLI below plays the agent: while the limit holds it writes a partial
``candidate.py`` and a stray file, edits the brief, and ends on the limit
message; once lifted it writes the real candidate.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import src.runtime.coding_agent as coding_agent
from src.runtime import token_usage
from src.runtime.usage_limits import (
    AgentLoginFailed,
    AgentUsageLimitExceeded,
    detect_login_failure,
    detect_usage_limit,
)

LA = ZoneInfo("America/Los_Angeles")

STUB = r'''
import json, sys
from pathlib import Path

calls_file, agent_dir, n_limited, backend, message = sys.argv[1:6]
calls = int(Path(calls_file).read_text()) + 1 if Path(calls_file).exists() else 1
Path(calls_file).write_text(str(calls))
agent_dir = Path(agent_dir)
limited = n_limited == "forever" or calls <= int(n_limited)

def emit(event):
    print(json.dumps(event), flush=True)

if limited:
    (agent_dir / "candidate.py").write_text("import pymc as pm\nmodel = pm.Mod")
    (agent_dir / "stray_notes.md").write_text("half done")
    (agent_dir / "CONTEXT.md").write_text("overwritten by the agent")
    (agent_dir / "scratch" / "explore.py").write_text("print('half')")
    if backend == "claude":
        emit({"type": "assistant", "message": {"content": [{"type": "text", "text": message}]}})
        emit({"type": "result", "subtype": "success", "result": message,
              "total_cost_usd": 0.5, "num_turns": 10,
              "usage": {"input_tokens": 10, "output_tokens": 5}})
        sys.exit(0)
    emit({"type": "error", "error": {"name": "APIError",
          "data": {"message": message, "statusCode": 429, "isRetryable": True}}})
    sys.exit(1)

(agent_dir / "candidate.py").write_text("FULL CANDIDATE")
(agent_dir / "hypothesis.md").write_text("a hypothesis")
if backend == "claude":
    # The agent's own words may mention limits; only the CLI's channels count.
    emit({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "You've hit your session limit is what the last agent saw."}]}})
    emit({"type": "result", "subtype": "success", "result": "Wrote candidate.py.",
          "total_cost_usd": 1.0, "num_turns": 7,
          "usage": {"input_tokens": 100, "output_tokens": 50}})
else:
    emit({"type": "text", "part": {"text": "Wrote candidate.py."}})
    emit({"type": "step_finish", "part": {"tokens": {"input": 100, "output": 50}, "cost": 0.1}})
'''


def claude_limit_message(hours_ahead: float = 1.0) -> str:
    """Claude's message, naming a reset ``hours_ahead`` from now (Pacific)."""
    reset = datetime.now(LA) + timedelta(hours=hours_ahead)
    clock = reset.strftime("%I:%M%p").lstrip("0").lower()
    return f"You've hit your session limit · resets {clock} (America/Los_Angeles)"


@pytest.fixture
def agent(tmp_path, monkeypatch):
    """A candidate dir with its brief, a stub CLI in the sandbox's place, and
    recorded waits instead of sleeps."""
    agent_dir = tmp_path / "candidate_0"
    agent_dir.mkdir()
    (agent_dir / "CONTEXT.md").write_text("the brief")
    stub = tmp_path / "stub_cli.py"
    stub.write_text(STUB)
    waits: list[float] = []
    monkeypatch.setattr(coding_agent, "_wait", waits.append)
    token_usage.reset_usage_log()
    yield {"dir": agent_dir, "stub": stub, "waits": waits, "calls": tmp_path / "calls"}
    token_usage.reset_usage_log()


def run_stub(tmp_path, monkeypatch, agent, *, backend, n_limited, message):
    def fake_sandbox_command(cmd, *, agent_dir, env, **kwargs):
        (agent_dir / "scratch").mkdir(exist_ok=True)
        (agent_dir / ".home").mkdir(exist_ok=True)
        return [
            sys.executable, str(agent["stub"]), str(agent["calls"]), str(agent_dir),
            str(n_limited), backend, message,
        ], dict(env)

    monkeypatch.setattr(coding_agent, "sandbox_command", fake_sandbox_command)
    return coding_agent.run_coding_agent(
        "p", cwd=tmp_path, log_path=agent["dir"] / "agent.jsonl", backend=backend,
        allowed_dirs=[agent["dir"]], writable_dirs=[agent["dir"]], sandbox=True,
        on_summary=None, timeout_secs=60, usage_label="inner:candidate",
    )


def assert_only_the_real_candidate(agent_dir: Path) -> None:
    assert (agent_dir / "candidate.py").read_text() == "FULL CANDIDATE"
    assert not (agent_dir / "stray_notes.md").exists()
    assert (agent_dir / "CONTEXT.md").read_text() == "the brief"


def test_claude_session_limit_is_waited_out_and_the_agent_run_again(tmp_path, monkeypatch, agent):
    success, result = run_stub(
        tmp_path, monkeypatch, agent, backend="claude", n_limited=3,
        message=claude_limit_message(hours_ahead=1),
    )
    assert (success, result) == (True, "Wrote candidate.py.")
    assert int(agent["calls"].read_text()) == 4
    # Each wait runs to the stated reset (about an hour) plus the margin.
    assert len(agent["waits"]) == 3
    assert all(55 * 60 < w < 65 * 60 for w in agent["waits"])
    assert_only_the_real_candidate(agent["dir"])
    assert not (agent["dir"] / "scratch" / "explore.py").exists()
    records = token_usage.records_since(0)
    assert [r.usage_limit is not None for r in records] == [True, True, True, False]
    assert all(r.usage_limit.startswith("You've hit your session limit") for r in records[:3])
    summary = token_usage.summarize(records)
    assert summary["n_usage_limit_hits"] == 3
    assert summary["usage_limit_wait_sec"] == pytest.approx(sum(agent["waits"]))
    log = (agent["dir"] / "agent.jsonl").read_text()
    assert log.count("--- usage limit; waiting") == 3


def test_a_limit_that_never_lifts_raises_after_the_maximum_wait(tmp_path, monkeypatch, agent):
    monkeypatch.setattr(coding_agent, "AGENT_USAGE_LIMIT_MAX_WAIT_SEC", 3 * 600)
    monkeypatch.setattr(coding_agent, "AGENT_USAGE_LIMIT_FALLBACK_WAIT_SEC", 600)
    with pytest.raises(AgentUsageLimitExceeded, match="did not lift within"):
        run_stub(
            tmp_path, monkeypatch, agent, backend="claude", n_limited="forever",
            message="You've reached your usage limit.",  # no reset time: fixed waits
        )
    assert agent["waits"] == [600, 600, 600]
    # Nothing the limited runs left behind can pass for a candidate.
    assert not (agent["dir"] / "candidate.py").exists()
    assert not (agent["dir"] / "stray_notes.md").exists()
    assert (agent["dir"] / "CONTEXT.md").read_text() == "the brief"


def test_a_reset_beyond_the_maximum_wait_raises_at_once(tmp_path, monkeypatch, agent):
    with pytest.raises(AgentUsageLimitExceeded):
        run_stub(
            tmp_path, monkeypatch, agent, backend="claude", n_limited="forever",
            message="You've hit your weekly limit · resets in 3 days 2 hours",
        )
    assert agent["waits"] == []
    assert not (agent["dir"] / "candidate.py").exists()


def test_api_rate_limit_waits_the_fixed_interval(tmp_path, monkeypatch, agent):
    success, _ = run_stub(
        tmp_path, monkeypatch, agent, backend="claude", n_limited=1,
        message='API Error: 429 {"type":"error","error":{"type":"rate_limit_error"}}',
    )
    assert success
    assert agent["waits"] == [coding_agent.AGENT_USAGE_LIMIT_FALLBACK_WAIT_SEC]
    assert_only_the_real_candidate(agent["dir"])


def test_opencode_quota_error_event_is_waited_out(tmp_path, monkeypatch, agent):
    success, result = run_stub(
        tmp_path, monkeypatch, agent, backend="opencode", n_limited=2,
        message="Resource has been exhausted (e.g. check quota).",
    )
    assert (success, result) == (True, "Wrote candidate.py.")
    assert len(agent["waits"]) == 2
    assert_only_the_real_candidate(agent["dir"])


def test_an_exhausted_api_credit_balance_raises_without_waiting(tmp_path, monkeypatch, agent):
    with pytest.raises(AgentLoginFailed, match="Credit balance is too low"):
        run_stub(
            tmp_path, monkeypatch, agent, backend="claude", n_limited="forever",
            message="Credit balance is too low",
        )
    assert agent["waits"] == []
    assert not (agent["dir"] / "candidate.py").exists()


def test_tools_that_requeue_on_limits_still_get_the_result_back(tmp_path, monkeypatch, agent):
    def fake_sandbox_command(cmd, *, agent_dir, env, **kwargs):
        return [sys.executable, str(agent["stub"]), str(agent["calls"]), str(agent_dir),
                "forever", "claude", "You've hit your session limit · resets 3pm"], dict(env)

    monkeypatch.setattr(coding_agent, "sandbox_command", fake_sandbox_command)
    (agent["dir"] / "scratch").mkdir()
    success, result = coding_agent.run_coding_agent(
        "p", cwd=tmp_path, log_path=agent["dir"] / "agent.jsonl", backend="claude",
        allowed_dirs=[agent["dir"]], writable_dirs=[agent["dir"]], sandbox=True,
        on_summary=None, timeout_secs=60, wait_out_usage_limits=False,
    )
    assert result.startswith("You've hit your session limit")
    assert agent["waits"] == []


def test_a_candidate_slot_spends_no_retry_on_a_usage_limit(tmp_path, monkeypatch, agent):
    """Through the candidate launcher: the slot's first attempt succeeds."""
    from src.pipelines.inner_loop.candidate_agent import _spawn_candidate_agent

    message = claude_limit_message(hours_ahead=0.5)

    def fake_sandbox_command(cmd, *, agent_dir, env, **kwargs):
        (agent_dir / "scratch").mkdir(exist_ok=True)
        return [sys.executable, str(agent["stub"]), str(agent["calls"]), str(agent_dir),
                "2", "claude", message], dict(env)

    monkeypatch.setattr(coding_agent, "sandbox_command", fake_sandbox_command)
    responses = tmp_path / "data" / "responses.csv"
    responses.parent.mkdir()
    responses.write_text("sequence_a,sequence_b,participant_id,trial_index,chose_left\n")
    ok = _spawn_candidate_agent(
        agent["dir"],
        {"context": "c", "brief": "b", "existing_hypotheses": "h"},
        models_dir=tmp_path / "models",
        responses_path=responses,
        agent_timeout_sec=60,
        backend="claude",
        agent_root=tmp_path,
    )
    assert ok
    assert len(agent["waits"]) == 2
    assert_only_the_real_candidate(agent["dir"])


def test_critique_round_does_not_swallow_a_usage_limit(tmp_path, monkeypatch):
    from src.pipelines.inner_loop import critique_round, scoring

    monkeypatch.setattr(scoring, "_best_exportable_model", lambda posterior, comparison: "seed")

    def limited(*args, **kwargs):
        raise AgentUsageLimitExceeded("limit")

    monkeypatch.setattr(critique_round, "_spawn_critique_agent", limited)
    with pytest.raises(AgentUsageLimitExceeded):
        critique_round._run_critique_round(
            tmp_path / "iter_0",
            responses_path=tmp_path / "responses.csv",
            models_dir=tmp_path / "models",
            posterior={},
            comparison={},
            cache_dir=None,
            fit_kwargs={},
            n_proposals=1,
            significance_alpha=0.05,
            n_replicates=1,
            agent_timeout_sec=1,
            backend="claude",
        )


@pytest.mark.parametrize(
    "message, reset",
    [
        # The two messages of the 2026-09-28 logs.
        ("You've hit your session limit · resets 2:20pm (America/Los_Angeles)",
         datetime(2026, 9, 28, 14, 20, tzinfo=LA)),
        ("You've hit your session limit · resets 11:20pm (America/Los_Angeles)",
         datetime(2026, 9, 28, 23, 20, tzinfo=LA)),
        # A reset that has just passed is this one, not tomorrow's.
        ("You've hit your session limit · resets 1:55pm (America/Los_Angeles)",
         datetime(2026, 9, 28, 13, 55, tzinfo=LA)),
        ("Claude AI usage limit reached|1790636400",
         datetime.fromtimestamp(1790636400).astimezone()),
        ("You've hit your usage limit. Upgrade to Pro or try again in 2 hours 5 minutes.",
         datetime(2026, 9, 28, 16, 5, tzinfo=LA)),
        ("You've hit your usage limit. Try again in 3 days 1 hour.",
         datetime(2026, 10, 1, 15, 0, tzinfo=LA)),
        ('API Error: 529 {"type":"error","error":{"type":"overloaded_error"}}', None),
        ("API Error: Repeated 529 Overloaded errors", None),
        ('{"code": 429, "status": "RESOURCE_EXHAUSTED"}', None),
        ("stream error: exceeded retry limit, last status: 429 Too Many Requests", None),
    ],
)
def test_limit_messages_and_their_reset_times(message, reset):
    now = datetime(2026, 9, 28, 14, 0, tzinfo=LA)
    limit = detect_usage_limit([message], now=now)
    assert limit is not None
    assert limit.reset_at == reset


def test_ordinary_output_is_not_a_limit():
    assert detect_usage_limit(["Wrote candidate.py.", "", "[bwrap] note"]) is None
    assert detect_login_failure(["Wrote candidate.py."]) is None
    assert detect_login_failure(["Invalid API key · Please run /login"]) is not None


def test_claude_limit_is_read_only_from_the_start_of_the_result_text():
    class Stream:
        final_result = "x" * 250 + " You've hit your session limit · resets 3pm"

    assert detect_usage_limit(coding_agent._cli_messages("claude", Stream(), "")) is None
    events = json.dumps({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "API Error: 429 rate_limit_error"}]}})
    Stream.final_result = "Done."
    assert detect_usage_limit(coding_agent._cli_messages("claude", Stream(), events)) is None
