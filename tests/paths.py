"""Repo paths and the standalone-script loader that the test suite shares.

Forty-six path constants had been re-derived across the suite (24 spellings of
the repo root alone, plus nine of the PyMC-model fixture dir), and fourteen
modules had each copied the same importlib loader for the standalone analysis
scripts. Both live here now, in the shared-module style of
``monitor_fixtures``/``viewer_fixtures``: one definition to keep correct if a
directory ever moves.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Optional

# Re-exported, not recomputed: `src.runtime.config` is the one repo root in the
# codebase (pyprojroot's here()), and a test suite that derived its own would be
# the tenth spelling of it. Test modules may import it from either place.
from src.runtime.config import REPO_ROOT

TESTS_DIR = REPO_ROOT / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"
# Hand-written PyMC cognitive models used as a stand-in model set.
PYMC_MODEL_FIXTURES_DIR = FIXTURES_DIR / "pymc_models"

SCRIPTS_DIR = REPO_ROOT / "scripts"
ANALYSIS_SCRIPTS_DIR = SCRIPTS_DIR / "analysis"


@lru_cache(maxsize=None)
def load_script_module(path: Path, name: Optional[str] = None) -> ModuleType:
    """Import a standalone script as a module so its helpers can be unit-tested.

    Several analysis entry points (``scripts/``, ``analysis/``) are scripts
    rather than importable packages, so the only way to reach their helpers is
    to load the file.

    Call this from inside a fixture or a test — never at module scope. At module
    scope the script executes during *collection*, which makes an unrelated
    script's import error or slow top-level work fail the whole suite's
    collection instead of just its own tests.

    Results are cached, so a script is imported at most once per session (what
    the module-scope loads used to give) even when several tests ask for it.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No script to load at {path}")
    spec = importlib.util.spec_from_file_location(name or path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because these scripts define their tyro
    # ``Args`` dataclass at import time, and @dataclass resolves string
    # annotations through ``sys.modules[cls.__module__]``.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
