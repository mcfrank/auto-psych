"""Loading a project's ``featurize_stimulus`` from its ``preprocess.py``.

A project supplies ``preprocess.py`` alongside its other assets; the pipeline
loads it **by file path** (project assets are not importable as packages) to
turn raw stimulus fields — e.g. a pair of H/T sequences — into the numeric
feature columns its PyMC models read via ``pm.Data``.

Three call sites (data collection, EIG, the outer-loop orchestrator) used to
hand-roll this loader with three different failure policies; two of them
returned ``None`` when the module loaded but exposed no ``featurize_stimulus``,
which quietly sent unfeaturized rows downstream. This is the one loader, and
every failure mode raises. Callers for whom "this project does not featurize"
is a legitimate state check that themselves (there is no ``preprocess.py``)
before calling.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict

# featurize_stimulus(sequence_a, sequence_b) -> {feature column: value}
Featurizer = Callable[[str, str], Dict[str, Any]]


def load_featurizer(featurize_path: Path) -> Featurizer:
    """Return ``featurize_stimulus`` from the module file at ``featurize_path``."""
    featurize_path = Path(featurize_path)
    if not featurize_path.exists():
        raise FileNotFoundError(f"featurize module not found: {featurize_path}")

    # The module name embeds the owning directory so two projects' preprocess.py
    # cannot clobber each other in sys.modules, and is registered before
    # exec_module so the module is importable by name while it executes.
    module_name = f"_featurizer_{featurize_path.parent.name}_{featurize_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, featurize_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load featurize module from {featurize_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    featurize = getattr(module, "featurize_stimulus", None)
    if featurize is None:
        raise AttributeError(f"{featurize_path} has no featurize_stimulus()")
    return featurize
