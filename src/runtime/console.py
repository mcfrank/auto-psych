"""Coherent console status messages for pipeline runs and agents."""

import sys


def log_status(msg: str, indent: bool = True) -> None:
    """Print a status line (indented by default for sub-steps)."""
    prefix = "    " if indent else ""
    print(prefix + msg, file=sys.stderr, flush=True)
