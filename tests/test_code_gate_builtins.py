"""Agent-written code cannot reach files or the interpreter behind the gate.

Candidate models and critique statistics are imported and run inside the
harness process, which can see the ground truth's files and its own command
line. The import allowlist alone let through ``open(...)``, ``__import__``,
``np.load``, ``eval`` and dunder escapes such as ``().__class__.__subclasses__()``;
a model has no need for any of them.
"""

from __future__ import annotations

import pytest

from src.pipelines.inner_loop.import_gate import check_forbidden_imports

CLEAN = """
import numpy as np
import pymc as pm
import re

PATTERN = re.compile("HT+")

def compute_features(sequence_a, sequence_b):
    return {"n": float(len(sequence_a))}
"""


def test_ordinary_model_code_passes():
    assert check_forbidden_imports(CLEAN) == []


@pytest.mark.parametrize(
    "snippet, flagged",
    [
        ("text = open('/proc/self/cmdline').read()", "open"),
        ("os = __import__('os')", "__import__"),
        ("x = eval('1+1')", "eval"),
        ("exec('y = 2')", "exec"),
        ("g = globals()", "globals"),
        ("import numpy as np\nd = np.load('/scratch/gt.npy')", "load"),
        ("import numpy as np\nd = np.loadtxt('f.csv')", "loadtxt"),
        ("s = ().__class__.__subclasses__()", "__subclasses__"),
        ("def f():\n    return f.__globals__", "__globals__"),
    ],
)
def test_file_and_interpreter_access_is_rejected(snippet, flagged):
    assert any(flagged in reason for reason in check_forbidden_imports(snippet))


# Allowed modules re-export forbidden ones, pandas (injected into critique
# statistics) reads files, and string formatting and getattr look attributes
# up where an AST check cannot see them. The harness's argv held
# `--gt-model <GT>`: `typing.sys.argv` passed the gate (audit, 2026-09-26).
@pytest.mark.parametrize(
    "snippet, flagged",
    [
        ("import typing\nargs = typing.sys.argv", "sys"),
        ("import dataclasses\nf = dataclasses.builtins.open('x')", "builtins"),
        ("import numpy as np\nf = np.DataSource(None).open('x')", "DataSource"),
        ("x = pd.read_csv('/scratch/gt.csv')", "read_csv"),
        ("x = pd.io.parsers", ".io"),
        ("from typing import sys", "use of sys"),
        ("from numpy import load", "use of load"),
        ("import numpy.lib.npyio as n", "numpy.lib.npyio"),
        ("from numpy.lib.npyio import DataSource", "numpy.lib.npyio"),
        ("import typing\ns = getattr(typing, 'sy' + 's')", "getattr"),
        (
            "import operator, typing\ns = operator.attrgetter('sys')(typing)",
            "attrgetter",
        ),
        ("import typing\ns = '{0.sys.argv}'.format(typing)", "attribute field"),
        ("import typing\nfmt = '{0.sys}'\ns = fmt.format(typing)", "attribute field"),
        (
            "try:\n    1 / 0\nexcept Exception as e:\n    g = e.__traceback__.tb_frame.f_back.f_globals",
            "__traceback__",
        ),
        ("d = (1).__class__.__base__.__dict__", "__dict__"),
    ],
)
def test_escapes_through_allowed_modules_are_rejected(snippet, flagged):
    assert any(flagged in reason for reason in check_forbidden_imports(snippet))


def test_ordinary_formatting_and_dataframe_code_passes():
    source = (
        "import numpy as np\n"
        "def test_statistic(df):\n"
        "    label = '{:.3f} of {}'.format(0.5, 2)\n"
        "    other = f'{len(df)} rows'\n"
        "    runs = df.groupby('sequence_a')['chose_left'].mean().to_numpy()\n"
        "    return float(np.mean(runs))\n"
    )
    assert check_forbidden_imports(source) == []


def test_the_harness_forgets_its_command_line_once_parsed(monkeypatch):
    """Its arguments name the held-out ground truth (--gt-model, run<r>/<gt>/
    paths), and agent-written code runs in its process."""
    import sys

    from src.subjective_randomness.holdout_recovery import forget_command_line

    argv = [
        "holdout_recovery.py",
        "--gt-model",
        "motif_stack",
        "--out",
        "run1/motif_stack/h.json",
    ]
    monkeypatch.setattr(sys, "argv", list(argv))
    monkeypatch.setattr(sys, "orig_argv", ["/venv/bin/python", *argv])
    seen_elsewhere = sys.argv  # e.g. typing.sys.argv: the same list

    forget_command_line()

    assert sys.argv == ["holdout_recovery.py"] and seen_elsewhere is sys.argv
    assert sys.orig_argv == ["/venv/bin/python", "holdout_recovery.py"]


@pytest.mark.parametrize(
    "script", ["holdout_recovery.py", "impossible_holdout_recovery.py"]
)
def test_the_harness_clis_forget_their_command_line_before_running(script):
    from tests.paths import SCRIPTS_DIR

    text = (SCRIPTS_DIR / "subjective_randomness" / script).read_text(encoding="utf-8")
    entry = text.split('if __name__ == "__main__":', 1)[1]
    assert entry.index("forget_command_line()") < entry.index("main(parsed)")
