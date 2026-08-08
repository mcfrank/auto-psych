"""Backend-agnostic launcher for coding agents (Claude Code or opencode).

Both pipeline loops spawn a coding-agent CLI as a subprocess, stream its
output to a log, and read back a (success, result) pair. The only thing that
differs between Claude Code and opencode is how the command line is built and
how the streamed output is interpreted. This module owns that difference so the
call sites stay backend-neutral.

Backend selection: an explicit argument wins, else the ``CODING_AGENT``
environment variable, else ``"opencode"`` (the default). Model names are
per-backend: each backend has its own default and any ``model`` argument is
passed through verbatim (opencode uses ``provider/model`` — defaults to Gemini;
Claude uses ``claude-sonnet-4-6``).

Both backends stream JSON events (Claude via ``--output-format stream-json``,
opencode via ``--format json``), and both report token usage in that stream —
Claude in the terminal ``result`` event, opencode in per-step ``step_finish``
events. Every run records its usage to :mod:`src.runtime.token_usage` under
the caller's ``usage_label`` so a pipeline run can account for its total
token spend.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.runtime import token_usage

DEFAULT_BACKEND = "opencode"
_DEFAULT_MODEL = {
    "claude": "claude-sonnet-4-6",
    "opencode": "google/gemini-3.1-pro-preview",
}

# opencode >= 1.17 keeps its sessions in one shared sqlite database. When
# several opencode instances start at once (parallel candidate agents), the
# late arrivals can die instantly with "database is locked" during session
# creation. That is transient — retry with backoff instead of losing the
# agent's whole task. Other failures are NOT retried.
OPENCODE_LOCK_RETRIES = 3
OPENCODE_LOCK_BACKOFF_SECS = 2.0
_OPENCODE_LOCK_SIGNATURE = "database is locked"


def select_backend(explicit: Optional[str]) -> str:
    """Resolve the coding-agent backend: explicit arg, then env, then default."""
    backend = explicit or os.environ.get("CODING_AGENT") or DEFAULT_BACKEND
    if backend not in _DEFAULT_MODEL:
        raise ValueError(
            f"unknown coding-agent backend: {backend!r} "
            f"(expected one of {sorted(_DEFAULT_MODEL)})"
        )
    return backend


def build_command(
    backend: str,
    *,
    prompt: str,
    allowed_dirs: list[Path],
    model: Optional[str],
) -> list[str]:
    """Build the CLI argv for the given backend.

    The prompt is always the final element so callers can locate it. opencode
    has no ``--add-dir`` equivalent (it operates on the working directory), so
    ``allowed_dirs`` is honoured only for Claude Code.
    """
    if backend not in _DEFAULT_MODEL:
        raise ValueError(f"unknown coding-agent backend: {backend!r}")
    model = model or _DEFAULT_MODEL[backend]
    if backend == "claude":
        cmd = [
            "claude",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        for d in allowed_dirs:
            cmd += ["--add-dir", str(d)]
        cmd += ["--model", model, "-p", prompt]
        return cmd
    if backend == "opencode":
        return ["opencode", "run", "--format", "json", "-m", model, prompt]
    # Reachable only if _DEFAULT_MODEL gains a backend without a branch here.
    # Fail loudly rather than returning None into subprocess.Popen.
    raise ValueError(f"no command builder for coding-agent backend: {backend!r}")


def _summarise_claude_event(event: dict) -> Optional[str]:
    """One-line human summary of a Claude stream-json event, or None to skip."""
    t = event.get("type")
    if t == "assistant":
        parts = event.get("message", {}).get("content", [])
        lines = []
        for part in parts if isinstance(parts, list) else []:
            if part.get("type") == "tool_use":
                name = part.get("name", "?")
                inp = part.get("input", {})
                detail = ""
                for key in ("command", "file_path", "pattern", "path", "query"):
                    if key in inp:
                        val = str(inp[key])
                        detail = f" {val[:120]}" if len(val) > 120 else f" {val}"
                        break
                lines.append(f"  → {name}{detail}")
            elif part.get("type") == "text":
                text = part.get("text", "").strip()
                if text:
                    lines.append(f"  … {text.splitlines()[0][:120]}")
        return "\n".join(lines) if lines else None
    if t == "result":
        subtype = event.get("subtype", "")
        cost = event.get("total_cost_usd", event.get("cost_usd"))
        cost_str = f"  cost=${cost:.4f}" if cost is not None else ""
        turns = event.get("num_turns", "?")
        result_text = str(event.get("result", ""))[:200]
        return f"  [result] {subtype}{cost_str}  turns={turns}\n  {result_text}"
    return None


def _summarise_opencode_event(event: dict) -> Optional[str]:
    """One-line human summary of an opencode ``--format json`` event, or None."""
    t = event.get("type")
    part = event.get("part", {})
    if t == "text":
        text = str(part.get("text", "")).strip()
        return f"  … {text.splitlines()[0][:120]}" if text else None
    if t == "tool":
        name = part.get("tool", part.get("name", "?"))
        return f"  → {name}"
    if t == "step_finish":
        tokens = part.get("tokens", {})
        cost = part.get("cost")
        cost_str = f"  cost=${cost:.4f}" if cost is not None else ""
        return f"  [step] tokens={tokens.get('total', '?')}{cost_str}"
    return None


class _ClaudeStream:
    """Interprets Claude Code stream-json events: result text, success, usage.

    Usage comes from the terminal ``result`` event (cumulative for the whole
    run). If the run dies before emitting it (timeout, crash), fall back to
    summing the per-call ``usage`` of the assistant events seen so far, so the
    tokens already paid for still get counted.
    """

    def __init__(self) -> None:
        self.final_result = ""
        self.success = False
        self._result_usage: Optional[Dict[str, Any]] = None
        self._result_cost: Optional[float] = None
        self._assistant_usages: list[Dict[str, Any]] = []

    def feed(self, event: dict) -> None:
        t = event.get("type")
        if t == "assistant":
            usage = event.get("message", {}).get("usage")
            if isinstance(usage, dict):
                self._assistant_usages.append(usage)
        elif t == "result":
            self.final_result = str(event.get("result", ""))
            self.success = event.get("subtype") == "success"
            usage = event.get("usage")
            if isinstance(usage, dict):
                self._result_usage = usage
            self._result_cost = event.get("total_cost_usd", event.get("cost_usd"))

    def usage_fields(self) -> Dict[str, Any]:
        if self._result_usage is not None:
            usages = [self._result_usage]
        elif self._assistant_usages:
            usages = self._assistant_usages
        else:
            return {"usage_missing": True}
        # Anthropic counts reasoning inside output_tokens and cached prompt
        # tokens outside input_tokens, so the components are already disjoint.
        return {
            "input_tokens": sum(int(u.get("input_tokens", 0)) for u in usages),
            "output_tokens": sum(int(u.get("output_tokens", 0)) for u in usages),
            "cache_write_tokens": sum(
                int(u.get("cache_creation_input_tokens", 0)) for u in usages
            ),
            "cache_read_tokens": sum(
                int(u.get("cache_read_input_tokens", 0)) for u in usages
            ),
            "cost_usd": self._result_cost,
        }


class _OpencodeStream:
    """Interprets opencode ``--format json`` events: result text and usage.

    Each ``step_finish`` event reports that step's tokens (disjoint components:
    input / output / reasoning / cache); a run's usage is their sum. The result
    text is the concatenation of the ``text`` parts.
    """

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._token_sums = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        self._cost = 0.0
        self._saw_usage = False

    def feed(self, event: dict) -> None:
        t = event.get("type")
        part = event.get("part", {})
        if t == "text":
            text = str(part.get("text", ""))
            if text.strip():
                self._texts.append(text)
        elif t == "step_finish":
            tokens = part.get("tokens")
            if isinstance(tokens, dict):
                self._saw_usage = True
                cache = tokens.get("cache", {})
                self._token_sums["input_tokens"] += int(tokens.get("input", 0))
                self._token_sums["output_tokens"] += int(tokens.get("output", 0))
                self._token_sums["reasoning_tokens"] += int(
                    tokens.get("reasoning", 0)
                )
                self._token_sums["cache_read_tokens"] += int(cache.get("read", 0))
                self._token_sums["cache_write_tokens"] += int(cache.get("write", 0))
            cost = part.get("cost")
            if cost is not None:
                self._cost += float(cost)

    def result_text(self) -> str:
        return "\n".join(self._texts).strip()

    def usage_fields(self) -> Dict[str, Any]:
        if not self._saw_usage:
            return {"usage_missing": True}
        return {**self._token_sums, "cost_usd": self._cost}


def run_coding_agent(
    prompt: str,
    *,
    cwd: Path,
    log_path: Path,
    allowed_dirs: Optional[list[Path]] = None,
    model: Optional[str] = None,
    timeout_secs: int = 900,
    backend: Optional[str] = None,
    env: Optional[dict] = None,
    on_summary: Optional[Callable[[str], None]] = print,
    usage_label: str = "coding_agent",
) -> tuple[bool, str]:
    """Spawn the selected coding agent, stream output to ``log_path``.

    Returns ``(success, result_text)``. For Claude, success and the final
    result come from the terminal ``result`` stream-json event; for opencode
    success is a zero exit code and the result is the text parts of its JSON
    event stream (falling back to raw output if no events parsed). On timeout
    returns ``(False, <message>)``.

    An opencode start that dies with "database is locked" (concurrent agents
    contending for opencode's shared sqlite store) is retried up to
    ``OPENCODE_LOCK_RETRIES`` times with backoff; any other failure is final.

    Whatever usage the stream reported is recorded to
    :mod:`src.runtime.token_usage` under ``usage_label`` — also on timeout or
    failure, since those tokens were spent all the same. One logical call
    records exactly one usage entry, from the attempt that ran.
    """
    backend = select_backend(backend)
    model = model or _DEFAULT_MODEL[backend]
    cmd = build_command(
        backend, prompt=prompt, allowed_dirs=list(allowed_dirs or []), model=model
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1 + OPENCODE_LOCK_RETRIES):
        outcome = _run_agent_once(
            cmd,
            backend=backend,
            cwd=cwd,
            log_path=log_path,
            timeout_secs=timeout_secs,
            env=env,
            on_summary=on_summary,
            log_mode="w" if attempt == 0 else "a",
        )
        stream, captured, timed_out, returncode = outcome
        raw_output = "".join(captured)
        lock_hit = (
            backend == "opencode"
            and not timed_out
            and returncode != 0
            and _OPENCODE_LOCK_SIGNATURE in raw_output
        )
        if lock_hit and attempt < OPENCODE_LOCK_RETRIES:
            delay = OPENCODE_LOCK_BACKOFF_SECS * (attempt + 1)
            message = (
                f"  [oc] opencode session store locked (attempt {attempt + 1}); "
                f"retrying in {delay:.0f}s"
            )
            if on_summary:
                on_summary(message)
            time.sleep(delay)
            continue
        break

    usage = stream.usage_fields()
    token_usage.record_usage(
        source=usage_label, backend=backend, model=model, **usage
    )
    if usage.get("usage_missing") and on_summary:
        on_summary(
            f"  [tokens] WARNING: {backend} run for {usage_label!r} reported no "
            f"token usage; this run's spend is uncounted"
        )

    if timed_out:
        return False, f"coding agent ({backend}) timed out after {timeout_secs}s"

    if backend == "claude":
        return stream.success, stream.final_result
    success = returncode == 0
    final_result = stream.result_text() or raw_output.strip()
    return success, final_result


def _run_agent_once(
    cmd: list[str],
    *,
    backend: str,
    cwd: Path,
    log_path: Path,
    timeout_secs: int,
    env: Optional[dict],
    on_summary: Optional[Callable[[str], None]],
    log_mode: str,
) -> tuple[Any, list[str], bool, Optional[int]]:
    """One subprocess pass: spawn, stream, kill on timeout.

    Returns ``(stream, captured_lines, timed_out, returncode)``. Retry attempts
    append to the log file (``log_mode="a"``) so the evidence of earlier
    failures survives.
    """
    stream = _ClaudeStream() if backend == "claude" else _OpencodeStream()
    summarise = (
        _summarise_claude_event if backend == "claude" else _summarise_opencode_event
    )
    captured: list[str] = []

    with open(log_path, log_mode, encoding="utf-8") as log_file:
        if log_mode == "a":
            log_file.write("\n--- retry attempt ---\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # Put the agent in its own process group so a timeout can kill the
            # whole tree. The coding-agent CLIs (claude/opencode/npx) spawn their
            # own children; proc.kill() would SIGKILL only the direct child, leak
            # the grandchildren, and — because a leaked grandchild can hold the
            # stdout pipe open — let the read loop / proc.wait() below hang forever.
            start_new_session=True,
        )
        timed_out = threading.Event()

        def _kill_after():
            timed_out.set()
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()

        timer = threading.Timer(timeout_secs, _kill_after)
        timer.start()
        try:
            for raw_line in proc.stdout:
                log_file.write(raw_line)
                log_file.flush()
                captured.append(raw_line)
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if on_summary:
                        prefix = "cc" if backend == "claude" else "oc"
                        on_summary(f"  [{prefix}] {line}")
                    continue
                if not isinstance(event, dict):
                    continue
                stream.feed(event)
                summary = summarise(event)
                if summary and on_summary:
                    on_summary(summary)
        finally:
            timer.cancel()
            proc.wait()

    return stream, captured, timed_out.is_set(), proc.returncode
