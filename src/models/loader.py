"""
Resolve model callables from run-directory .py files or the built-in MODEL_LIBRARY.
"""

import importlib.util
from pathlib import Path
from typing import Callable, List, Optional

from src.models.randomness import MODEL_LIBRARY

# Callable type: (stimulus, response_options) -> dict[str, float]
ModelCallable = Callable[..., dict]


def get_model_callable(model_name: str, theorist_dir: Optional[Path] = None) -> ModelCallable:
    """
    Return the model function for the given name.
    If theorist_dir is set and contains <model_name>.py, load and return that module's
    callable (function with same name as the file, e.g. bayesian_fair_coin).
    Otherwise return MODEL_LIBRARY[model_name]. Raises if not found in either place.
    """
    theorist_dir = Path(theorist_dir) if theorist_dir else None
    py_path = (theorist_dir / f"{model_name}.py") if theorist_dir else None

    if py_path and py_path.exists():
        spec = importlib.util.spec_from_file_location(
            f"theorist_model_{model_name}", py_path, submodule_search_locations=[]
        )
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"Cannot load module from {py_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Prefer a function with the same name as the file (with underscores)
        fn = getattr(mod, model_name, None)
        if callable(fn):
            return fn
        # Otherwise first callable that looks like (stimulus, response_options) -> dict
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if callable(obj):
                return obj
        raise ValueError(f"No callable found in {py_path} (expected function '{model_name}')")

    if model_name in MODEL_LIBRARY:
        return MODEL_LIBRARY[model_name]
    raise KeyError(f"Model '{model_name}' not in run dir ({py_path}) and not in MODEL_LIBRARY")


def get_model_names_from_manifest(manifest: dict, theorist_dir: Optional[Path] = None) -> List[str]:
    """
    Return list of model names from manifest that are loadable: either
    <name>.py exists in theorist_dir or name is in MODEL_LIBRARY.
    """
    theorist_dir = Path(theorist_dir) if theorist_dir else None
    models = manifest.get("models") or []
    names = []
    for m in models:
        name = m.get("name") if isinstance(m, dict) else (m if isinstance(m, str) else None)
        if not name:
            continue
        py_exists = theorist_dir and (theorist_dir / f"{name}.py").exists()
        if py_exists or name in MODEL_LIBRARY:
            names.append(name)
    return names
