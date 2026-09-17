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


def check_forbidden_imports(source: str) -> List[str]:
    """Return top-level module names of imports not in the allowlist.

    Walks the full AST (including nested function bodies) so an import hidden
    inside ``def compute_features`` is still caught. Relative imports (``from .
    import ...``) are always forbidden — agent code has no package context.
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
    return sorted(set(forbidden))
