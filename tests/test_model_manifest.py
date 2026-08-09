"""Tests for the on-disk model manifests and the one reader that parses them.

``models_manifest.yaml`` names the active model set of a directory of PyMC
models. Two of these manifests are project assets rather than run outputs:

* the **registry** (``src/subjective_randomness/pymc_model_families/``) — the
  literature-faithful seed set, and the single source of truth for which
  models are active;
* the **live seed pool**
  (``src/pipelines/outer_loop/projects/subjective_randomness/seed_models/``) —
  what the outer loop copies into experiment 1.

The two drifted apart once already (the hero-run winners were promoted into
the seed pool in 2026-07, the faithful set was consolidated into the registry
in 2026-08), which silently broke every recovery helper that resolves a seed
name to its pure-Python twin. The tests here are the landmine that makes such
a divergence fail loudly at the next test run.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.models.model_manifest import (
    MANIFEST_FILENAME,
    read_loadable_model_names,
    read_manifest_entries,
    read_manifest_names,
)
from tests.paths import REPO_ROOT

REGISTRY_DIR = REPO_ROOT / "src" / "subjective_randomness" / "pymc_model_families"
LIVE_SEED_DIR = (
    REPO_ROOT
    / "src"
    / "pipelines"
    / "outer_loop"
    / "projects"
    / "subjective_randomness"
    / "seed_models"
)


@pytest.mark.parametrize(
    "models_dir", [REGISTRY_DIR, LIVE_SEED_DIR], ids=["registry", "live_seed_pool"]
)
def test_every_manifest_model_has_a_pure_python_twin(models_dir):
    """Each active model resolves to ``model_families.<name>.DEFAULT_PARAMS``.

    Recovery needs the twin: ``model_recovery.default_generating_params``
    imports it by name to fix the generating parameters, and the twin is what
    the PyMC adapter is validated against. A manifest name without a twin is a
    ``ModuleNotFoundError`` waiting for the next recovery run.
    """
    names = read_manifest_names(models_dir)
    assert names, f"{models_dir} has an empty manifest"
    for name in names:
        family = importlib.import_module(
            f"src.subjective_randomness.model_families.{name}"
        )
        assert dict(family.DEFAULT_PARAMS), f"{name} has empty DEFAULT_PARAMS"


@pytest.mark.parametrize(
    "models_dir", [REGISTRY_DIR, LIVE_SEED_DIR], ids=["registry", "live_seed_pool"]
)
def test_every_manifest_model_has_its_pymc_file(models_dir):
    for name in read_manifest_names(models_dir):
        assert (models_dir / f"{name}.py").exists()


# ── the reader ──────────────────────────────────────────────────────


def _write_manifest(models_dir: Path, body: str) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / MANIFEST_FILENAME).write_text(body, encoding="utf-8")
    return models_dir


def test_read_manifest_entries_keeps_order_and_rationales(tmp_path):
    _write_manifest(
        tmp_path,
        "models:\n"
        "  - name: second_listed\n    rationale: People do B.\n"
        "  - name: first_listed\n    rationale: People do A.\n",
    )
    assert read_manifest_entries(tmp_path) == [
        {"name": "second_listed", "rationale": "People do B."},
        {"name": "first_listed", "rationale": "People do A."},
    ]
    assert read_manifest_names(tmp_path) == ["second_listed", "first_listed"]


def test_read_manifest_entries_normalises_bare_string_entries(tmp_path):
    _write_manifest(tmp_path, "models:\n  - just_a_name\n")
    assert read_manifest_entries(tmp_path) == [{"name": "just_a_name"}]


def test_read_manifest_entries_accepts_an_empty_model_list(tmp_path):
    # A freshly created zoo directory legitimately lists no models yet; only
    # callers that require a non-empty set say so themselves.
    _write_manifest(tmp_path, "models: []\n")
    assert read_manifest_entries(tmp_path) == []


def test_missing_manifest_raises_and_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match=MANIFEST_FILENAME):
        read_manifest_names(tmp_path)


def test_missing_manifest_is_an_empty_set_only_when_asked(tmp_path):
    assert read_manifest_entries(tmp_path, missing_ok=True) == []


def test_malformed_yaml_raises_naming_the_path(tmp_path):
    _write_manifest(tmp_path, "models:\n  - name: a\n   rationale: bad indent\n")
    with pytest.raises(ValueError, match=MANIFEST_FILENAME):
        read_manifest_entries(tmp_path)


def test_non_mapping_manifest_raises(tmp_path):
    _write_manifest(tmp_path, "- name: a\n")
    with pytest.raises(ValueError, match="mapping"):
        read_manifest_entries(tmp_path)


def test_entry_without_a_name_raises(tmp_path):
    # Silently skipping it would drop a model from the set without a trace.
    _write_manifest(tmp_path, "models:\n  - rationale: People do A.\n")
    with pytest.raises(ValueError, match="no name"):
        read_manifest_entries(tmp_path)


def test_read_loadable_model_names_keeps_only_models_with_a_file(tmp_path):
    _write_manifest(tmp_path, "models:\n  - name: present\n  - name: ghost\n")
    (tmp_path / "present.py").write_text("", encoding="utf-8")
    assert read_loadable_model_names(tmp_path) == ["present"]
