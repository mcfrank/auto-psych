"""
Shared observability for pipeline agents: a timestamped per-agent log file.

Every agent writes to <agent_dir>/observability.log so you can see what the
agent did, including validation feedback on retries.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

AGENT_LOG_FILENAME = "observability.log"


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
