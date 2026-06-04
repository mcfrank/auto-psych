"""
Shared observability for pipeline agents: timestamped log file and LLM transcripts.

Every agent writes to <agent_dir>/observability.log and can write transcripts to
<agent_dir>/transcripts/ so you can see what the agent did and exactly what the LLM
sent and received (including validation feedback on retries).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

AGENT_LOG_FILENAME = "observability.log"
TRANSCRIPTS_DIRNAME = "transcripts"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def agent_log(out_dir: Path, *lines: str) -> None:
    """Append timestamped lines to observability.log in out_dir. Creates file if needed."""
    if not out_dir:
        return
    out_dir = Path(out_dir)
    log_path = out_dir / AGENT_LOG_FILENAME
    with open(log_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"[{_ts()}] {line}\n")
        f.flush()


def append_validation_failure(
    run_dir: Path,
    agent_key: str,
    attempt: int,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a validation failure line to the agent's observability.log (called by validator)."""
    agent_dir = Path(run_dir) / agent_key
    agent_dir.mkdir(parents=True, exist_ok=True)
    line = f"Validation failed (attempt {attempt}): {message}"
    if details:
        line += f" | details={details}"
    agent_log(agent_dir, line)


def write_transcript(
    out_dir: Path,
    attempt: int,
    *,
    system: str = "",
    user: str = "",
    response: str = "",
    validation_feedback: str = "",
) -> Path:
    """
    Write one transcript file for this attempt to out_dir/transcripts/attempt_NNN.md.
    Includes system prompt, user message, full LLM response, and any validation feedback.
    Returns the path to the written file.
    """
    out_dir = Path(out_dir)
    transcripts_dir = out_dir / TRANSCRIPTS_DIRNAME
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    path = transcripts_dir / f"attempt_{attempt:03d}.md"
    parts = [
        "# LLM transcript",
        f"Attempt: {attempt}",
        f"Recorded: {_ts()}",
        "",
    ]
    if validation_feedback:
        parts.extend(["## Validation feedback (previous attempt)", "", validation_feedback, ""])
    if system:
        parts.extend(["## System prompt", "", system, ""])
    if user:
        parts.extend(["## User message", "", user, ""])
    if response:
        parts.extend(["## LLM response", "", response, ""])
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
