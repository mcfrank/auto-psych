"""Resolve model callables from a theorist run directory."""

import importlib.util
from pathlib import Path
from typing import Callable, List, Optional

# Callable type: (stimulus, response_options) -> dict[str, float]
ModelCallable = Callable[..., dict]


def get_model_callable(
    model_name: str, theorist_dir: Optional[Path] = None
) -> ModelCallable:
    """
    Return the model function for the given name from the theorist run dir only.
    theorist_dir must be set and must contain <model_name>.py. Raises if not found.
    (For ground truth, use get_ground_truth_models(project_id) from src.models.project.ground_truth.)
    """
    theorist_dir = Path(theorist_dir) if theorist_dir else None
    if not theorist_dir:
        raise KeyError(
            f"theorist_dir required to load model '{model_name}' (no global library)"
        )
    py_path = theorist_dir / f"{model_name}.py"
    if not py_path.exists():
        raise FileNotFoundError(f"Model '{model_name}' has no {py_path}")

    spec = importlib.util.spec_from_file_location(
        f"theorist_model_{model_name}", py_path, submodule_search_locations=[]
    )
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, model_name, None)
    if callable(fn):
        return fn
    # No function named after the model. Fall back ONLY when the module defines
    # exactly one public callable of its own — refusing to guess among several
    # (which could silently bind prediction/validation to the wrong symbol) or to
    # return an imported helper (filter by __module__). Otherwise fail loudly.
    own_callables = [
        getattr(mod, name)
        for name in dir(mod)
        if not name.startswith("_")
        and callable(getattr(mod, name))
        and getattr(getattr(mod, name), "__module__", None) == mod.__name__
    ]
    if len(own_callables) == 1:
        return own_callables[0]
    raise ValueError(
        f"{py_path} defines no function named '{model_name}' and "
        f"{'several' if own_callables else 'no'} public module-level callables were "
        f"found; name the model's entry-point function '{model_name}'."
    )


def get_model_names_from_manifest(
    manifest: dict, theorist_dir: Optional[Path] = None
) -> List[str]:
    """
    Return model names from manifest that are loadable: <name>.py exists in theorist_dir.
    No global library fallback; we only use the theorist's run outputs.
    """
    theorist_dir = Path(theorist_dir) if theorist_dir else None
    models = manifest.get("models") or []
    names = []
    for m in models:
        name = (
            m.get("name")
            if isinstance(m, dict)
            else (m if isinstance(m, str) else None)
        )
        if not name:
            continue
        if theorist_dir and (theorist_dir / f"{name}.py").exists():
            names.append(name)
    return names
