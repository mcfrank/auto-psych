"""Guard: nothing in the pipeline or holdout path imports src.subjective_randomness.features.

After P9 raw is the only mode: the feature library is a research-only tool.
The pipeline, models, critique, holdout harness, seed sets and model families
must work without it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from pyprojroot import here

REPO = here()

GUARDED_TREES = [
    REPO / "src" / "pipelines",
    REPO / "src" / "models",
    REPO / "src" / "critique",
    REPO / "src" / "subjective_randomness" / "model_families",
]

GUARDED_FILES = [
    REPO / "src" / "subjective_randomness" / "holdout_recovery.py",
]

SEED_DIRS = [
    REPO / "src" / "pipelines" / "outer_loop" / "projects" / "subjective_randomness" / "seed_models",
    REPO / "src" / "subjective_randomness" / "pymc_model_families",
]

FORBIDDEN_MODULES = {
    "src.subjective_randomness.features",
    "subjective_randomness.features",
}

# Research analysis tools that legitimately need the feature library.
EXCLUDED_FILES = {
    REPO / "src" / "pipelines" / "outer_loop" / "projects" / "subjective_randomness" / "evaluate_recovery.py",
}


def _python_files(root: Path):
    if root.is_file() and root.suffix == ".py":
        if root not in EXCLUDED_FILES:
            yield root
    elif root.is_dir():
        for p in root.rglob("*.py"):
            if p not in EXCLUDED_FILES:
                yield p


def _ast_imports_feature_module(path: Path) -> list[str]:
    """Return import statements that reference the features module."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.endswith(m) or alias.name == m for m in FORBIDDEN_MODULES):
                    hits.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(mod.endswith(m) or mod == m for m in FORBIDDEN_MODULES):
                hits.append(f"from {mod} import ...")
            if mod == ".." or mod == "..features" or mod.endswith("..features"):
                hits.append(f"from {mod} import ...")
    # Also check relative imports: from .. import features
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.names:
                for alias in node.names:
                    if alias.name == "features":
                        hits.append(f"from {'.'*node.level}{node.module or ''} import features")
    return hits


class TestASTNoFeatureImports:
    """No AST-level import of src.subjective_randomness.features."""

    @pytest.mark.parametrize("root", GUARDED_TREES + SEED_DIRS, ids=lambda p: str(p.relative_to(REPO)))
    def test_guarded_trees(self, root):
        violations = {}
        for py in _python_files(root):
            hits = _ast_imports_feature_module(py)
            if hits:
                violations[str(py.relative_to(REPO))] = hits
        assert not violations, f"Feature imports found:\n{violations}"

    @pytest.mark.parametrize("path", GUARDED_FILES, ids=lambda p: str(p.relative_to(REPO)))
    def test_guarded_files(self, path):
        hits = _ast_imports_feature_module(path)
        assert not hits, f"Feature imports in {path.relative_to(REPO)}:\n{hits}"


class TestSubprocessNoFeatureImports:
    """Importing with features poisoned must succeed."""

    ENTRY_MODULES = [
        "src.pipelines.outer_loop.orchestrator",
        "src.pipelines.inner_loop.pymc_orchestrator",
        "src.subjective_randomness.holdout_recovery",
    ]

    @pytest.mark.parametrize("module", ENTRY_MODULES)
    def test_import_with_features_poisoned(self, module):
        code = (
            "import sys; "
            "sys.modules['src.subjective_randomness.features'] = None; "
            f"import {module}"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Importing {module} with features poisoned failed:\n"
            f"stderr: {result.stderr}"
        )
