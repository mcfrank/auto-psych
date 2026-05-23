"""Backend-agnostic launcher for coding agents (Claude Code or opencode).

Both pipeline loops spawn a coding-agent CLI as a subprocess, stream its
output to a log, and read back a (success, result) pair. The only thing that
differs between Claude Code and opencode is how the command line is built and
how the streamed output is interpreted. This module owns that difference so the
call sites stay backend-neutral.

Backend selection: an explicit argument wins, else the ``CODING_AGENT``
environment variable, else ``"claude"``. Model names are per-backend: each
backend has its own default and any ``model`` argument is passed through
verbatim (Claude uses ``claude-sonnet-4-6``; opencode uses ``provider/model``).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

DEFAULT_BACKEND = "claude"
_DEFAULT_MODEL = {
    "claude": "claude-sonnet-4-6",
    "opencode": "anthropic/claude-sonnet-4-6",
}


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
            "--output-format", "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        for d in allowed_dirs:
            cmd += ["--add-dir", str(d)]
        cmd += ["--model", model, "-p", prompt]
        return cmd
    if backend == "opencode":
        return ["opencode", "run", "-m", model, prompt]


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
        cost = event.get("cost_usd")
        cost_str = f"  cost=${cost:.4f}" if cost is not None else ""
        turns = event.get("num_turns", "?")
        result_text = str(event.get("result", ""))[:200]
        return f"  [result] {subtype}{cost_str}  turns={turns}\n  {result_text}"
    return None


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
) -> tuple[bool, str]:
    """Spawn the selected coding agent, stream output to ``log_path``.

    Returns ``(success, result_text)``. For Claude, success and the final
    result come from the terminal ``result`` stream-json event; for opencode
    (plain stdout) success is a zero exit code and the result is the full
    captured output. On timeout returns ``(False, <message>)``.
    """
    backend = select_backend(backend)
    cmd = build_command(
        backend, prompt=prompt, allowed_dirs=list(allowed_dirs or []), model=model
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    final_result = ""
    success = False
    captured: list[str] = []

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        timed_out = threading.Event()

        def _kill_after():
            timed_out.set()
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
                if backend == "claude":
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        if on_summary:
                            on_summary(f"  [cc] {line}")
                        continue
                    summary = _summarise_claude_event(event)
                    if summary and on_summary:
                        on_summary(summary)
                    if event.get("type") == "result":
                        final_result = str(event.get("result", ""))
                        success = event.get("subtype") == "success"
                elif on_summary:
                    on_summary(f"  [oc] {line}")
        finally:
            timer.cancel()
            proc.wait()

    if timed_out.is_set():
        return False, f"coding agent ({backend}) timed out after {timeout_secs}s"

    if backend != "claude":
        success = proc.returncode == 0
        final_result = "".join(captured).strip()
    return success, final_result
