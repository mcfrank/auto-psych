"""Static AST check: reject agent-written code that imports outside an allowlist.

Every candidate model and critique test statistic must be self-contained. The
allowlist covers the numeric/modelling stack the agents need (numpy, pymc,
pytensor, arviz, scipy) plus selected stdlib modules. Anything else — especially
the project's feature library — is rejected at admission before the code is ever
executed, and the rejection is recorded in the hypothesis ledger.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

CANDIDATE_IMPORT_ALLOWLIST = frozenset(
    {
        "numpy",
        "pymc",
        "pytensor",
        "arviz",
        "scipy",
        "math",
        "itertools",
        "functools",
        "collections",
        "re",
        "typing",
        "dataclasses",
        "statistics",
        "operator",
    }
)


# Names and attributes agent code may not use. The code runs inside the harness
# process, which can see the ground truth's files and its own command line; the
# import allowlist alone let file reads (open, np.load), dynamic imports and
# interpreter escapes through, and a model needs none of them.
FORBIDDEN_NAMES = frozenset(
    {"open", "__import__", "exec", "eval", "compile", "globals", "vars",
     "breakpoint", "input", "__builtins__"}
)
FORBIDDEN_ATTRIBUTES = frozenset(
    {"load", "loadtxt", "genfromtxt", "fromfile", "memmap",
     "__globals__", "__builtins__", "__subclasses__", "__code__", "__getattribute__"}
)


def check_forbidden_imports(source: str) -> List[str]:
    """Return why ``source`` may not run: forbidden imports and forbidden uses.

    Imports outside the allowlist are reported by module name; uses of
    ``FORBIDDEN_NAMES`` (a bare ``open``, ``eval``, ...) and
    ``FORBIDDEN_ATTRIBUTES`` (``np.load``, ``__subclasses__``, ...) as
    "use of X". Walks the full AST (including nested function bodies) so an
    import hidden inside ``def compute_features`` is still caught. Relative
    imports (``from . import ...``) are always forbidden — agent code has no
    package context.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["<unparseable source>"]
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in CANDIDATE_IMPORT_ALLOWLIST:
                    forbidden.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                forbidden.append(f"relative import (level {node.level})")
                continue
            if node.module:
                top = node.module.split(".")[0]
                if top not in CANDIDATE_IMPORT_ALLOWLIST:
                    forbidden.append(node.module)
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            forbidden.append(f"use of {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            forbidden.append(f"use of .{node.attr}")
    return sorted(set(forbidden))
