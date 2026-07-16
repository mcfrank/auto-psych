"""Load project-specific ground-truth models."""

import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.runtime.config import project_dir


def get_ground_truth_models(
    project_id: str,
) -> Dict[str, Callable[..., Dict[str, float]]]:
    """
    Load GROUND_TRUTH_MODELS from projects/<project_id>/ground_truth_models.py.
    Returns name -> callable (stimulus, response_options) -> dict.

    An absent module legitimately means "this project has no ground-truth
    models" and returns {}. A module that exists but fails to import or lacks
    a GROUND_TRUTH_MODELS dict is a broken project asset and raises — silently
    treating it as "no models" would make a ground-truth run collect from the
    wrong generator.
    """
    proj = project_dir(project_id)
    path = proj / "ground_truth_models.py"
    if not path.exists():
        return {}
    spec = importlib.util.spec_from_file_location(
        f"ground_truth_models_{project_id}", path, submodule_search_locations=[]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    registry = getattr(mod, "GROUND_TRUTH_MODELS", None)
    if not isinstance(registry, dict):
        raise ValueError(
            f"{path} exists but defines no GROUND_TRUTH_MODELS dict "
            f"(got {type(registry).__name__})."
        )
    return dict(registry)


def get_ground_truth_model_names(project_id: str) -> List[str]:
    """Return list of ground-truth model names for the project (for CLI validation)."""
    return list(get_ground_truth_models(project_id).keys())
