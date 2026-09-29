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


@pytest.mark.parametrize("snippet, flagged", [
    ("text = open('/proc/self/cmdline').read()", "open"),
    ("os = __import__('os')", "__import__"),
    ("x = eval('1+1')", "eval"),
    ("exec('y = 2')", "exec"),
    ("g = globals()", "globals"),
    ("import numpy as np\nd = np.load('/scratch/gt.npy')", "load"),
    ("import numpy as np\nd = np.loadtxt('f.csv')", "loadtxt"),
    ("s = ().__class__.__subclasses__()", "__subclasses__"),
    ("def f():\n    return f.__globals__", "__globals__"),
])
def test_file_and_interpreter_access_is_rejected(snippet, flagged):
    assert any(flagged in reason for reason in check_forbidden_imports(snippet))
