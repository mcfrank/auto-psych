"""
Load ground-truth models from a project. Used only for --ground-truth-model.
Projects define their own ground_truth_models.py with GROUND_TRUTH_MODELS dict.
"""

import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.config import project_dir


def get_ground_truth_models(project_id: str) -> Dict[str, Callable[..., Dict[str, float]]]:
    """
    Load GROUND_TRUTH_MODELS from projects/<project_id>/ground_truth_models.py.
    Returns name -> callable (stimulus, response_options) -> dict.
    Returns {} if the module does not exist or has no GROUND_TRUTH_MODELS.
    """
    proj = project_dir(project_id)
    path = proj / "ground_truth_models.py"
    if not path.exists():
        return {}
    try:
        spec = importlib.util.spec_from_file_location(
            f"ground_truth_models_{project_id}", path, submodule_search_locations=[]
        )
        if spec is None or spec.loader is None:
            return {}
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        registry = getattr(mod, "GROUND_TRUTH_MODELS", None)
        if isinstance(registry, dict):
            return dict(registry)
    except Exception:
        pass
    return {}


def get_ground_truth_model_names(project_id: str) -> List[str]:
    """Return list of ground-truth model names for the project (for CLI validation)."""
    return list(get_ground_truth_models(project_id).keys())
