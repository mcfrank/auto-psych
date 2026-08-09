"""Coherent console status messages for pipeline runs and agents."""

import sys
from typing import Optional

AGENT_DISPLAY_NAMES = {
    "2_design": "Design",
    "3_implement": "Implement",
    "4_collect": "Collect data",
    "5_model_loop": "Model loop",
}


def agent_header(
    agent_key: str,
    run_id: int,
    total_runs: Optional[int] = None,
    mode: Optional[str] = None,
) -> None:
    """Print agent section header with run context and optional mode."""
    name = AGENT_DISPLAY_NAMES.get(agent_key, agent_key)
    if total_runs is not None:
        run_ctx = f"Run {run_id}/{total_runs}"
    else:
        run_ctx = f"Run {run_id}"
    parts = [f"#### {name} agent ({run_ctx})"]
    if mode:
        parts.append(f"mode={mode}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def log_status(msg: str, indent: bool = True) -> None:
    """Print a status line (indented by default for sub-steps)."""
    prefix = "    " if indent else ""
    print(prefix + msg, file=sys.stderr, flush=True)
