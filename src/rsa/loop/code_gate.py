"""What agent-written memo model files may import and use.

The general denylist of `src.pipelines.inner_loop.import_gate` (file readers,
dynamic imports, introspection escapes) applies unchanged; only the allowed
packages differ from the PyMC domain's: the JAX stack memo models are written
in, and from this repository exactly one module, `src.rsa.memo_kit`.
"""

from __future__ import annotations

from typing import List

from src.pipelines.inner_loop.import_gate import check_forbidden_imports

RSA_IMPORT_ALLOWLIST = frozenset(
    {
        "jax",
        "memo",
        "numpyro",
        "numpy",
        "math",
        "itertools",
        "functools",
        "collections",
        "typing",
        "dataclasses",
        "enum",
        "operator",
    }
)
RSA_EXACT_MODULES = frozenset({"src.rsa.memo_kit"})


def code_problems(source: str) -> List[str]:
    """Why ``source`` may not run as an RSA model file; empty when it may."""
    return check_forbidden_imports(
        source, allowlist=RSA_IMPORT_ALLOWLIST, exact_modules=RSA_EXACT_MODULES
    )
