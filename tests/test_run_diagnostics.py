"""A long-running harness can be diagnosed while it runs: its log lines carry
the time, and SIGUSR1 dumps every thread's stack (no ptrace, which the cluster
nodes refuse for processes one did not start)."""

from __future__ import annotations

import re
import signal
import subprocess
import sys
import time

from tests.paths import REPO_ROOT, SCRIPTS_DIR

SNIPPET = """
import sys, time
from src.runtime.diagnostics import install_run_diagnostics
install_run_diagnostics()
print("first line", flush=True)
print("two\\nlines", flush=True)
print("to stderr", file=sys.stderr, flush=True)
sys.stdout.write("partial ")
sys.stdout.write("line\\n")
sys.stdout.flush()
print("ready", flush=True)
time.sleep(30)
"""


def test_log_lines_carry_the_time_and_sigusr1_dumps_every_threads_stack():
    proc = subprocess.Popen(
        [sys.executable, "-c", SNIPPET],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = []
    while True:
        line = proc.stdout.readline()
        lines.append(line)
        if "ready" in line or not line:
            break
    proc.send_signal(signal.SIGUSR1)
    time.sleep(1)
    proc.kill()
    out, err = proc.communicate()
    lines += out.splitlines(keepends=True)
    stamp = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
    assert re.match(stamp + "first line$", lines[0])
    assert re.match(stamp + "two$", lines[1]) and re.match(stamp + "lines$", lines[2])
    assert re.match(stamp + "partial line$", lines[3])
    assert re.search(stamp + "to stderr$", err, re.M)
    assert "Current thread" in err and "time.sleep" not in out  # the stack dump went to stderr


def test_the_harness_cli_installs_the_diagnostics_before_running():
    text = (SCRIPTS_DIR / "subjective_randomness" / "holdout_recovery.py").read_text(encoding="utf-8")
    entry = text.split('if __name__ == "__main__":', 1)[1]
    assert entry.index("install_run_diagnostics()") < entry.index("main(parsed)")
