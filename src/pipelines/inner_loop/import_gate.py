"""Static AST check: reject agent-written code that imports outside an allowlist.

Every candidate model and critique test statistic must be self-contained. The
allowlist covers the numeric/modelling stack the agents need (numpy, pymc,
pytensor, arviz, scipy) plus selected stdlib modules. Anything else — especially
the project's feature library — is rejected at admission before the code is ever
executed, and the rejection is recorded in the hypothesis ledger.
"""

from __future__ import annotations

import ast
import string
from pathlib import Path
from typing import AbstractSet, List

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
#
# Allowed modules re-export forbidden ones as attributes (``typing.sys``,
# ``dataclasses.builtins``, ``numpy.lib.npyio``), and pandas — injected into
# critique statistics — reads files (``pd.read_csv``), so the attribute list
# covers the module names, the file readers and writers of the allowed
# libraries, and the introspection routes (frames, tracebacks, ``getattr``,
# attribute paths inside ``str.format`` fields) that reach the rest. It is a
# denylist over the known routes, not a proof: the harness also forgets its
# command line once parsed (scripts/subjective_randomness/holdout_recovery.py).
FORBIDDEN_NAMES = frozenset(
    {
        "open",
        "__import__",
        "exec",
        "eval",
        "compile",
        "globals",
        "vars",
        "locals",
        "breakpoint",
        "input",
        "__builtins__",
        "getattr",
        "setattr",
        "delattr",
        "__loader__",
        "__spec__",
    }
)
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        # modules reachable as attributes of allowed ones
        "sys",
        "os",
        "builtins",
        "io",
        "subprocess",
        "pathlib",
        "shutil",
        "importlib",
        "ctypes",
        "ctypeslib",
        "socket",
        "pickle",
        "marshal",
        "npyio",
        "_datasource",
        "modules",
        # file readers and writers
        "open",
        "read",
        "write",
        "load",
        "loadtxt",
        "genfromtxt",
        "fromfile",
        "fromregex",
        "memmap",
        "open_memmap",
        "DataSource",
        "loadmat",
        "savemat",
        "from_netcdf",
        "from_json",
        "from_zarr",
        "load_arviz_data",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "tofile",
        "to_netcdf",
        "to_csv",
        "to_pickle",
        "to_parquet",
        "to_hdf",
        "to_sql",
        "to_excel",
        "to_feather",
        "to_stata",
        # getattr by another name
        "attrgetter",
        "methodcaller",
        # introspection escapes
        "__globals__",
        "__builtins__",
        "__subclasses__",
        "__code__",
        "__getattribute__",
        "__dict__",
        "__bases__",
        "__base__",
        "__mro__",
        "__traceback__",
        "__loader__",
        "__spec__",
        "__closure__",
        "__self__",
        "__func__",
        "__reduce__",
        "__reduce_ex__",
        "tb_frame",
        "tb_next",
        "f_back",
        "f_globals",
        "f_locals",
        "f_builtins",
        "f_code",
        "gi_frame",
        "cr_frame",
        "ag_frame",
    }
)
# Attribute prefixes: pandas' file readers (read_csv, read_json, read_pickle, ...).
FORBIDDEN_ATTRIBUTE_PREFIXES = ("read_",)


def _forbidden_attribute(attr: str) -> bool:
    return attr in FORBIDDEN_ATTRIBUTES or attr.startswith(FORBIDDEN_ATTRIBUTE_PREFIXES)


def _format_reaches_attributes(node: ast.Call) -> bool:
    """A ``str.format``/``format_map`` call that can look up attributes.

    A replacement field such as ``"{0.sys.argv}"`` makes ``format`` perform
    the attribute lookup at run time, where the AST check cannot see it. Only
    a call on a string literal whose fields index nothing is allowed.
    """
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr in ("format", "format_map")):
        return False
    if not (isinstance(func.value, ast.Constant) and isinstance(func.value.value, str)):
        return True
    try:
        fields = [
            field for _, field, _, _ in string.Formatter().parse(func.value.value)
        ]
    except ValueError:
        return True
    return any(field and ("." in field or "[" in field) for field in fields)


def check_forbidden_imports(
    source: str,
    *,
    allowlist: AbstractSet[str] = CANDIDATE_IMPORT_ALLOWLIST,
    exact_modules: AbstractSet[str] = frozenset(),
) -> List[str]:
    """Return why ``source`` may not run: forbidden imports and forbidden uses.

    Imports outside the allowlist are reported by module name; uses of
    ``FORBIDDEN_NAMES`` (a bare ``open``, ``eval``, ``getattr``, ...) and of
    ``FORBIDDEN_ATTRIBUTES`` (``np.load``, ``typing.sys``, ``pd.read_csv``,
    ``__subclasses__``, ...) — as an attribute or as a name imported with
    ``from … import`` — as "use of X"; a ``str.format`` that can look up
    attributes as "use of .format with an attribute field". Walks the full AST
    (including nested function bodies) so an import hidden inside
    ``def compute_features`` is still caught. Relative imports
    (``from . import ...``) are always forbidden — agent code has no package
    context.

    ``allowlist`` names the allowed top-level packages (default: the PyMC
    stack's); ``exact_modules`` allows single modules by full dotted name
    without opening their package (the RSA domain allows
    ``src.rsa.memo_kit``, never ``src``).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["<unparseable source>"]
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if alias.name in exact_modules:
                    continue
                if parts[0] not in allowlist:
                    forbidden.append(alias.name)
                elif any(_forbidden_attribute(part) for part in parts[1:]):
                    forbidden.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                forbidden.append(f"relative import (level {node.level})")
                continue
            if node.module and node.module not in exact_modules:
                parts = node.module.split(".")
                if parts[0] not in allowlist or any(
                    _forbidden_attribute(part) for part in parts[1:]
                ):
                    forbidden.append(node.module)
                    continue
            for alias in node.names:
                if alias.name in FORBIDDEN_NAMES or _forbidden_attribute(alias.name):
                    forbidden.append(f"use of {alias.name}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            forbidden.append(f"use of {node.id}")
        elif isinstance(node, ast.Attribute) and _forbidden_attribute(node.attr):
            forbidden.append(f"use of .{node.attr}")
        elif isinstance(node, ast.Call) and _format_reaches_attributes(node):
            forbidden.append(f"use of .{node.func.attr} with an attribute field")
    return sorted(set(forbidden))
