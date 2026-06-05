"""Shared config/path helpers for the subjective-randomness scripts."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict

import yaml
from pyprojroot import here

REPO_ROOT = here()


def resolve_path(path_value: str | Path, config_path: Path | None = None) -> Path:
    """Resolve a path relative to the repo root, falling back to the config dir.

    Absolute paths are returned unchanged. Relative paths are first interpreted
    against the repo root; if that location does not exist and a config path is
    given, they are resolved relative to the config file's directory instead.
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists() or config_path is None:
        return repo_path
    return (config_path.parent / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_model(module_path: str):
    return importlib.import_module(module_path)
