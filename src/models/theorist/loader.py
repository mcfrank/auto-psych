"""Resolve model callables from a theorist run directory."""

import importlib.util
import logging
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

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
        # Explicit, logged fallback: which symbol got bound decides every
        # prediction attributed to this model, so it must be visible in the run
        # log rather than inferred from behavior.
        fallback = own_callables[0]
        logger.warning(
            "%s defines no function named %r; binding its only public callable %r "
            "instead. Rename that function to %r to make the entry point explicit.",
            py_path,
            model_name,
            getattr(fallback, "__name__", repr(fallback)),
            model_name,
        )
        return fallback
    raise ValueError(
        f"{py_path} defines no function named '{model_name}' and "
        f"{'several' if own_callables else 'no'} public module-level callables were "
        f"found; name the model's entry-point function '{model_name}'."
    )


def get_model_names_from_manifest(
    manifest: dict, theorist_dir: Optional[Path] = None
) -> List[str]:
    """
    Return manifest model names after verifying every implementation exists.

    No global library fallback; the manifest defines the complete hypothesis
    set in ``theorist_dir``. Missing or malformed entries raise instead of
    shrinking that set.
    """
    if not isinstance(manifest, dict):
        raise ValueError("model manifest must be a mapping")
    models = manifest.get("models")
    if not isinstance(models, list):
        raise ValueError("model manifest must contain a 'models' list")
    if theorist_dir is None:
        raise ValueError("theorist_dir is required to resolve manifest model files")

    theorist_dir = Path(theorist_dir)
    names: List[str] = []
    for entry in models:
        name = entry.get("name") if isinstance(entry, dict) else entry
        if not isinstance(name, str) or not name:
            raise ValueError(f"invalid model manifest entry: {entry!r}")
        if name in names:
            raise ValueError(f"duplicate model name in manifest: {name!r}")
        names.append(name)

    missing = [name for name in names if not (theorist_dir / f"{name}.py").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Manifest lists model file(s) missing from {theorist_dir}: {missing}"
        )
    return names
