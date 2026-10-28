"""Make a long-running harness diagnosable while it runs.

``install_run_diagnostics`` stamps every line the process writes to stdout and
stderr with the wall-clock time, and registers SIGUSR1 to dump every thread's
Python stack to stderr (``kill -USR1 <pid>``). The cluster nodes refuse ptrace
for processes one did not start, so a profiler cannot attach to a running job.
Log output only: nothing the run computes changes.
"""

from __future__ import annotations

import faulthandler
import signal
import sys
import time
from typing import Any, TextIO


class _TimestampedStream:
    """A text stream that prefixes each line with the time it was started."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._at_line_start = True

    def write(self, text: str) -> int:
        out = []
        for piece in text.splitlines(keepends=True):
            if self._at_line_start:
                out.append(time.strftime("%Y-%m-%d %H:%M:%S "))
            out.append(piece)
            self._at_line_start = piece.endswith("\n")
        self._stream.write("".join(out))
        return len(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def install_run_diagnostics() -> None:
    """Timestamp this process's stdout and stderr lines; dump all threads'
    stacks to stderr on SIGUSR1."""
    faulthandler.register(signal.SIGUSR1, file=sys.__stderr__, all_threads=True)
    sys.stdout = _TimestampedStream(sys.stdout)
    sys.stderr = _TimestampedStream(sys.stderr)
