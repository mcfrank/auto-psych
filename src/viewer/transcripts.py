"""Clean the coding-agent terminal transcripts for display.

The ``*.jsonl`` files written by the outer-loop stages and per-candidate inner
loop are not JSON — they are raw colored terminal output from the coding agent.
This module turns them into plain, human-readable text.
"""

from __future__ import annotations

import re

# Matches CSI sequences (e.g. "\x1b[0m") and the bare "[0m"-style residue that
# shows up in the logs when the leading ESC byte was already stripped upstream.
_ANSI_CSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BARE_SGR = re.compile(r"\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from terminal output, keeping the text."""
    text = _ANSI_CSI.sub("", text)
    text = _BARE_SGR.sub("", text)
    return text
