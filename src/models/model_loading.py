"""PyMC model loading and caching.

Loads agent-written ``<name>.py`` files that define a module-level ``pm.Model``,
attaches optional data-binding hooks (``compute_features``, ``prepare_observed``),
and exposes a per-process model cache.

Lazy imports: ``pymc`` / ``arviz`` / ``pytensor`` are imported locally so that
importing this module is cheap when only cache utilities are used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List  # Any used for lazily-imported pymc types


# ---------------------------------------------------------------------------
# Hook attribute names attached to every loaded model
# ---------------------------------------------------------------------------

# Attribute under which a loaded model carries its ``compute_features`` hook.
_COMPUTE_FEATURES_ATTR = "_auto_psych_compute_features"

# Attribute under which a loaded model carries its optional data-preparation
# hook (a ``prepare_observed(rows) -> dict[str, np.ndarray]`` callable).
_PREPARE_OBSERVED_ATTR = "_auto_psych_prepare_observed"


# ---------------------------------------------------------------------------
# Lazy heavy-library imports
# ---------------------------------------------------------------------------

def _import_pymc() -> Any:
    """Lazy-import ``pymc``; avoids the cost when only cache utilities are used."""
    import pymc as pm

    return pm


def _import_arviz() -> Any:
    """Lazy-import ``arviz``."""
    import arviz as az

    return az


# ---------------------------------------------------------------------------
# Module execution
# ---------------------------------------------------------------------------

def _exec_model_module(py_path: Path, *, mod_prefix: str) -> Any:
    """Import a model `.py` as a standalone module and return the module object.

    Shared by :func:`load_pymc_model` (which then requires a module-level
    ``model``) and :func:`model_sampler_settings` (which only needs a
    module-level constant, and must work even for a file that builds no model).
    ``mod_prefix`` keeps the two callers' ``sys.modules`` entries distinct.
    """
    py_path = Path(py_path)
    if not py_path.exists():
        raise FileNotFoundError(f"PyMC model file not found: {py_path}")
    unique_mod_name = (
        f"{mod_prefix}{py_path.stem}_"
        f"{hashlib.sha1(str(py_path).encode()).hexdigest()[:8]}"
    )
    spec = importlib.util.spec_from_file_location(
        unique_mod_name, py_path, submodule_search_locations=[]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build module spec for {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_mod_name] = mod
    # Compile the source bytes ourselves instead of ``spec.loader.exec_module``,
    # which consults ``__pycache__``. importlib validates a cached ``.pyc`` by
    # (mtime, size) only, so a model file rewritten in the same second to the
    # same length — e.g. a candidate whose `0.9` became `0.8` — loads STALE
    # bytecode. Reading the source directly makes "loaded model" always mean
    # "what is on disk right now", and writes no bytecode cache to invalidate.
    source = py_path.read_bytes()
    exec(compile(source, str(py_path), "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pymc_model(name: str, models_dir: Path) -> Any:
    """Import ``models_dir/<name>.py`` and return its module-level ``pm.Model``.

    Fails loudly if the file is missing, fails to import, or does not expose a
    ``pm.Model`` at module level.
    """
    pm = _import_pymc()
    models_dir = Path(models_dir)
    py_path = models_dir / f"{name}.py"
    mod = _exec_model_module(py_path, mod_prefix="_pymc_model_")

    model = getattr(mod, "model", None)
    if not isinstance(model, pm.Model):
        raise TypeError(
            f"{py_path} must define a module-level `model: pm.Model` "
            f"(got {type(model).__name__ if model is not None else 'missing'})"
        )

    # A model declares ``compute_features(sequence_a, sequence_b) -> dict``
    # to derive numeric columns from raw H/T sequences. Attached to the model
    # so every data-binding path computes them before binding pm.Data.
    compute_features_fn = getattr(mod, "compute_features", None)
    if compute_features_fn is not None and not callable(compute_features_fn):
        raise TypeError(
            f"{py_path}: `compute_features` must be a callable "
            f"(sequence_a, sequence_b) -> dict, got {type(compute_features_fn).__name__}"
        )
    setattr(model, _COMPUTE_FEATURES_ATTR, compute_features_fn)

    # Optional model-owned data-preparation hook: a model may declare
    # ``prepare_observed(rows) -> dict[str, np.ndarray]`` to build its ``pm.Data``
    # arrays itself. The default convention maps one CSV column per container,
    # which cannot express layouts where the containers are not all trial-aligned
    # — e.g. motif_stack's unique-sequence table plus per-trial gather indices.
    # When declared, it REPLACES the column-mapping path entirely.
    prepare_observed = getattr(mod, "prepare_observed", None)
    if prepare_observed is not None and not callable(prepare_observed):
        raise TypeError(
            f"{py_path}: `prepare_observed` must be a callable "
            f"(rows) -> dict[str, np.ndarray], got {type(prepare_observed).__name__}"
        )
    if prepare_observed is not None and compute_features_fn is not None:
        raise ValueError(
            f"{py_path} declares BOTH `prepare_observed` and `compute_features`. "
            "They are alternative data-binding conventions — `prepare_observed` "
            "owns every container, so `compute_features` would be silently "
            "ignored. Declare exactly one."
        )
    setattr(model, _PREPARE_OBSERVED_ATTR, prepare_observed)
    return model


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------

def pm_data_inputs(model: Any) -> List[str]:
    """Return the names of every `pm.Data` container in the model."""
    from pytensor.tensor.sharedvar import TensorSharedVariable

    return [
        name
        for name, var in model.named_vars.items()
        if isinstance(var, TensorSharedVariable)
    ]


def observed_response_data(model: Any) -> str:
    """Return the name of the `pm.Data` container holding observed responses.

    Walks back from `model.observed_RVs` through the pytensor graph to find
    its `TensorSharedVariable` ancestor. Fails loudly if zero or more than
    one observed RV, or if its observed tensor has zero or multiple shared
    ancestors.
    """
    try:
        from pytensor.graph.traversal import ancestors
    except ImportError:  # pytensor < 2.31 kept it in graph.basic
        from pytensor.graph.basic import ancestors
    from pytensor.tensor.sharedvar import TensorSharedVariable

    if len(model.observed_RVs) == 0:
        raise ValueError(
            "Model has no observed RVs; cannot identify response data container."
        )
    if len(model.observed_RVs) > 1:
        raise ValueError(
            f"Model has {len(model.observed_RVs)} observed RVs; expected exactly one. "
            f"Got: {[rv.name for rv in model.observed_RVs]}"
        )

    rv = model.observed_RVs[0]
    obs_value = model.rvs_to_values.get(rv)
    if obs_value is None:
        raise ValueError(f"Observed RV {rv.name!r} has no observed value tensor.")

    shared = [a for a in ancestors([obs_value]) if isinstance(a, TensorSharedVariable)]
    if not shared:
        raise ValueError(
            f"Observed RV {rv.name!r} is not backed by a pm.Data container. "
            "Pass the pm.Data tensor directly to observed=."
        )
    if len(shared) > 1:
        names = [s.name for s in shared]
        raise ValueError(
            f"Observed RV {rv.name!r} traces back to multiple pm.Data containers: {names}. "
            "Pass exactly one pm.Data tensor to observed=."
        )
    return shared[0].name


# ---------------------------------------------------------------------------
# Per-process model cache
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[tuple, Any] = {}


def load_pymc_model_cached(name: str, models_dir: Path) -> Any:
    """Per-process cache of loaded PyMC models, keyed by (name, models_dir).

    Loading involves importlib + executing the model file's `with pm.Model()`
    block; cheap (no MCMC), but worth caching when called many times — e.g.
    EIG over hundreds of candidate stimuli.
    """
    key = (name, str(Path(models_dir).resolve()))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_pymc_model(name, Path(models_dir))
    return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    """Clear the loaded-model cache. Useful for tests."""
    _MODEL_CACHE.clear()
