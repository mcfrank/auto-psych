"""PyMC inference bridge for cognitive models.

The theorist agent writes each model as a `<name>.py` file with a module-level
`with pm.Model() as model:` block. This module loads those models, fits them
to observed data via MCMC, and exposes posterior-mean predictions, ELPD-LOO
for Bayesian model comparison, and posterior-predictive samples for PPC.

Convention: every `pm.Data` container in the model must have a name matching
a column in the preprocessed responses CSV. The bridge auto-pulls
`df[name].values` for each container. The observed-response container is
identified by tracing `model.observed_RVs[0]` back through the pytensor graph
to its `TensorSharedVariable` ancestor.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.mcmc_defaults import (
    PRODUCTION_CHAINS,
    PRODUCTION_CORES,
    PRODUCTION_DRAWS,
    PRODUCTION_TARGET_ACCEPT,
    PRODUCTION_TUNE,
)
from src.models.probability import validate_probability, validate_probability_array
from src.registry.io import validate_theory_weights

# Imports of pymc / arviz / pytensor are local in each function so that
# loading this module is cheap when only e.g. cache utilities are used.


# Attribute under which a loaded model carries its optional theorist-supplied
# featurizer (a ``compute_features(sequence_a, sequence_b) -> dict`` callable).
_EXTRA_FEATURIZER_ATTR = "_auto_psych_extra_featurizer"

# Attribute under which a loaded model carries its optional data-preparation
# hook (a ``prepare_observed(rows) -> dict[str, np.ndarray]`` callable).
_PREPARE_OBSERVED_ATTR = "_auto_psych_prepare_observed"


def _import_pymc():
    import pymc as pm

    return pm


def _import_arviz():
    import arviz as az

    return az


def _exec_model_module(py_path: Path, *, mod_prefix: str):
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


def load_pymc_model(name: str, models_dir: Path):
    """Import `models_dir/<name>.py` and return its module-level `model` attribute.

    Fails loudly if the file is missing, fails to import, or does not expose a
    `pm.Model` at module level.
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

    # Optional theorist-extensible featurizer: a model may declare
    # ``compute_features(sequence_a, sequence_b) -> dict[str, float]`` to add
    # numeric feature columns the base featurizer never produced (e.g.
    # order/position-sensitive statistics). We attach it to the model so every
    # data-binding path (extract_observed / make_stim_data) computes those
    # columns from the raw H/T sequences before binding pm.Data containers.
    featurizer = getattr(mod, "compute_features", None)
    if featurizer is not None and not callable(featurizer):
        raise TypeError(
            f"{py_path}: `compute_features` must be a callable "
            f"(sequence_a, sequence_b) -> dict, got {type(featurizer).__name__}"
        )
    setattr(model, _EXTRA_FEATURIZER_ATTR, featurizer)

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
    if prepare_observed is not None and featurizer is not None:
        raise ValueError(
            f"{py_path} declares BOTH `prepare_observed` and `compute_features`. "
            "They are alternative data-binding conventions — `prepare_observed` "
            "owns every container, so an extra featurizer would be silently "
            "ignored. Declare exactly one."
        )
    setattr(model, _PREPARE_OBSERVED_ATTR, prepare_observed)
    return model


def pm_data_inputs(model) -> List[str]:
    """Return the names of every `pm.Data` container in the model."""
    from pytensor.tensor.sharedvar import TensorSharedVariable

    return [
        name
        for name, var in model.named_vars.items()
        if isinstance(var, TensorSharedVariable)
    ]


def observed_response_data(model) -> str:
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


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with Path(csv_path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _model_extra_featurizer(model):
    """The model's optional ``compute_features`` callable, or ``None``."""
    return getattr(model, _EXTRA_FEATURIZER_ATTR, None)


def _augment_rows_with_features(
    model, rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Add a model's theorist-declared extra features to each row.

    If the model carries a ``compute_features(sequence_a, sequence_b)``
    featurizer, run it over every row's raw H/T sequences and merge the numeric
    columns it returns. A no-op (returns ``rows`` unchanged) for models that do
    not declare one. Fails loudly — never silently drops or coerces — if:

    - the featurizer is declared but the rows lack ``sequence_a``/``sequence_b``;
    - it returns something other than a dict, or a non-finite/non-numeric value;
    - it returns different feature names for different rows;
    - a returned feature name collides with an existing column.
    """
    featurizer = _model_extra_featurizer(model)
    if featurizer is None or not rows:
        return rows

    missing = {"sequence_a", "sequence_b"} - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Model declares compute_features but rows are missing "
            f"{sorted(missing)}; the raw H/T sequence columns are required to "
            "compute extra features."
        )

    augmented: List[Dict[str, Any]] = []
    expected_keys: Optional[tuple] = None
    for i, r in enumerate(rows):
        extra = featurizer(r["sequence_a"], r["sequence_b"])
        if not isinstance(extra, dict):
            raise TypeError(
                f"compute_features must return a dict of feature_name -> number, "
                f"got {type(extra).__name__} for row {i}."
            )
        keys = tuple(sorted(extra.keys()))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(
                "compute_features returned inconsistent feature names: row 0 -> "
                f"{list(expected_keys)}, row {i} -> {list(keys)}. It must return "
                "the same feature names for every stimulus."
            )
        for name, value in extra.items():
            if name in r:
                raise ValueError(
                    f"compute_features feature {name!r} collides with an existing "
                    "column; extra features must use new names."
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"compute_features feature {name!r} must be a number, got "
                    f"{type(value).__name__} ({value!r})."
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"compute_features feature {name!r} is not finite ({value!r})."
                )
        augmented.append({**r, **extra})
    return augmented


def _model_prepare_observed(model):
    """The model's optional ``prepare_observed`` callable, or ``None``."""
    return getattr(model, _PREPARE_OBSERVED_ATTR, None)


def _observed_via_hook(model, rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Build every ``pm.Data`` array through the model's ``prepare_observed`` hook.

    The hook owns the layout, so the harness cannot check it column by column.
    It checks the contract instead, and fails loudly on any breach — a hook that
    returned the wrong keys or a mis-shaped array would otherwise surface much
    later as an inscrutable pytensor shape error, or worse, as a silently
    mis-aligned likelihood:

    - the returned keys must be exactly the model's ``pm.Data`` names;
    - every value must be a numpy array whose rank matches the container's
      placeholder, and whose dtype has the same *kind* (a float array for an
      integer container would be truncated on binding);
    - the observed-response container must have one entry per input row, which
      is what makes ``p_left`` per-trial and keeps ELPD-LOO pointwise.

    Exact dtype *width* is normalized here rather than demanded of the hook,
    because the width is PyMC's choice, not the model's: ``pm.Data`` converts an
    ``int64`` array to ``intX`` (int32 under PyMC 5.28), so a hook that hard-coded
    a width would break on a different build. This mirrors what the
    column-mapping path does with ``placeholder.dtype``.
    """
    prepare = _model_prepare_observed(model)
    if prepare is None:
        raise ValueError("Model declares no prepare_observed hook.")
    out = prepare(list(rows))
    if not isinstance(out, dict):
        raise TypeError(
            "prepare_observed must return a dict of pm.Data name -> numpy array, "
            f"got {type(out).__name__}."
        )
    expected = set(pm_data_inputs(model))
    got = set(out)
    if got != expected:
        missing = sorted(expected - got)
        unexpected = sorted(got - expected)
        raise ValueError(
            "prepare_observed must return exactly the model's pm.Data containers. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )
    bound: Dict[str, np.ndarray] = {}
    for name in sorted(out):
        arr = out[name]
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"prepare_observed returned {type(arr).__name__} for {name!r}; "
                "every value must be a numpy array."
            )
        placeholder = model.named_vars[name].get_value()
        if arr.ndim != placeholder.ndim:
            raise ValueError(
                f"prepare_observed returned a {arr.ndim}-D array for {name!r} but "
                f"its pm.Data placeholder is {placeholder.ndim}-D."
            )
        if arr.dtype.kind != placeholder.dtype.kind:
            raise ValueError(
                f"prepare_observed returned dtype {arr.dtype} for {name!r} but its "
                f"pm.Data placeholder is {placeholder.dtype} — the kinds differ, so "
                "binding would silently reinterpret the values."
            )
        cast = arr.astype(placeholder.dtype, copy=False)
        if not np.array_equal(cast, arr):
            raise ValueError(
                f"prepare_observed's {name!r} array does not survive the cast to the "
                f"container's dtype {placeholder.dtype} (values out of range)."
            )
        bound[name] = cast
    out = bound
    response_name = observed_response_data(model)
    n_response = len(out[response_name])
    if n_response != len(rows):
        raise ValueError(
            f"prepare_observed returned {n_response} entries for the observed-response "
            f"container {response_name!r} but was given {len(rows)} rows; the observed "
            "response must stay one-per-trial."
        )
    return out


def make_stim_data(model, rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Build a `pm.set_data` dict from a list of row dicts for a given model.

    Each `pm.Data` container in `model` is filled with the corresponding column
    from `rows`, cast to the placeholder's dtype. Useful for predict_p_left and
    sample_synthetic_responses, where the caller has rows but not a CSV file.

    If the model declares a ``prepare_observed`` hook it builds every container
    itself and the column mapping is skipped. Otherwise, if the model declares a
    ``compute_features`` featurizer, its extra columns are computed from each
    row's raw sequences first.
    """
    if _model_prepare_observed(model) is not None:
        return _observed_via_hook(model, rows)
    rows = _augment_rows_with_features(model, rows)
    inputs = pm_data_inputs(model)
    missing = [c for c in inputs if rows and c not in rows[0]]
    if missing:
        raise ValueError(
            f"Rows missing columns {missing} required by the model. "
            f"Available: {list(rows[0].keys()) if rows else []}"
        )
    out: Dict[str, np.ndarray] = {}
    for col in inputs:
        placeholder = model.named_vars[col].get_value()
        dtype = placeholder.dtype
        values = [r[col] for r in rows]
        if np.issubdtype(dtype, np.integer):
            arr = np.array([int(float(v)) for v in values], dtype=dtype)
        elif np.issubdtype(dtype, np.floating):
            arr = np.array([float(v) for v in values], dtype=dtype)
        else:
            arr = np.array(values, dtype=dtype)
        out[col] = arr
    return out


def extract_observed(csv_path: Path, model) -> Dict[str, np.ndarray]:
    """Read csv_path and pull one numpy array per pm.Data container in the model.

    Dtype is inferred from the model's current pm.Data placeholder (int64,
    float64, etc.). Fails loudly if any expected column is missing.

    A model that declares a ``prepare_observed`` hook builds its containers from
    the raw CSV rows instead (see :func:`_observed_via_hook`).
    """
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    if _model_prepare_observed(model) is not None:
        return _observed_via_hook(model, rows)
    rows = _augment_rows_with_features(model, rows)

    inputs = pm_data_inputs(model)
    missing = [c for c in inputs if c not in rows[0]]
    if missing:
        raise ValueError(
            f"Responses CSV {csv_path} is missing columns {missing} required by the model. "
            f"Available columns: {list(rows[0].keys())}"
        )

    out: Dict[str, np.ndarray] = {}
    for col in inputs:
        placeholder = model.named_vars[col].get_value()
        dtype = placeholder.dtype
        values = [r[col] for r in rows]
        if np.issubdtype(dtype, np.integer):
            arr = np.array([int(float(v)) for v in values], dtype=dtype)
        elif np.issubdtype(dtype, np.floating):
            arr = np.array([float(v) for v in values], dtype=dtype)
        else:
            arr = np.array(values, dtype=dtype)
        out[col] = arr
    return out


# Errors that mean the *harness* (or the environment) is broken rather than
# "this model cannot be fit to this data". Reporting them through a
# (False, reason) screening contract would reject every candidate for a reason
# that has nothing to do with the candidates, so they propagate. Shared with
# ``src.pipelines.outer_loop.eig._screen_usable_models``, which screens models
# for the same kind of reason.
BROKEN_MODEL_CODE_ERRORS = (
    ImportError,
    SyntaxError,
    IndentationError,
    NameError,
    AttributeError,
)


def model_logp_is_finite(
    name: str, models_dir: Path, responses_path: Path
) -> tuple[bool, str]:
    """Fast, sampling-free check that a model can actually be MCMC-fit.

    Loads the model, binds the real responses, and evaluates the total log
    probability at the initial point. Returns ``(True, "")`` when that logp is
    finite, else ``(False, reason)``.

    A model whose graph evaluates to NaN or ``-inf`` — e.g. the numerically
    unsafe ``pt.sqrt(x**2)``, which NaNs in PyTensor for some inputs — passes
    graph-loading but crashes ``pm.sample`` at its start-value check, aborting
    the whole run. This catches such a model cheaply, before any sampling.

    NUTS also needs a finite *gradient* of the logp, and it evaluates the logp
    at jittered initial points. We therefore check both the logp and its gradient
    at the initial point. This is still not a full guarantee (a logp that only
    NaNs once NUTS jitters off the initial point can slip through), but it catches
    the common non-finite-gradient failure that a logp-only check misses.

    ``(False, reason)`` states one thing only: *this model cannot be fit to this
    data*. A broken harness (see ``BROKEN_MODEL_CODE_ERRORS``) is not that, and
    propagates — otherwise a missing dependency would silently condemn every
    candidate the inner loop generated.
    """
    pm = _import_pymc()
    model = load_pymc_model(name, models_dir)
    try:
        observed = extract_observed(responses_path, model)
        with model:
            pm.set_data(observed)
    except BROKEN_MODEL_CODE_ERRORS:
        raise
    except Exception as e:
        # A candidate (or seed) model that references feature columns the
        # responses don't carry — e.g. it declares extra pm.Data inputs without
        # a matching compute_features featurizer — is simply unfittable. Reject
        # it via this gate's (False, reason) contract so the caller drops/skips
        # it, rather than letting the error abort the whole inner loop.
        return False, f"cannot bind responses to model: {type(e).__name__}: {e}"
    try:
        point = model.initial_point()
        logp = float(model.compile_logp()(point))
    except BROKEN_MODEL_CODE_ERRORS:
        raise
    except Exception as e:  # a graph that cannot even be evaluated
        return False, f"logp evaluation raised: {type(e).__name__}: {e}"
    if not math.isfinite(logp):
        return False, f"non-finite logp ({logp}) at the initial point"
    try:
        grad = np.asarray(model.compile_dlogp()(point), dtype=float)
    except BROKEN_MODEL_CODE_ERRORS:
        raise
    except Exception as e:
        return False, f"gradient evaluation raised: {type(e).__name__}: {e}"
    if not np.all(np.isfinite(grad)):
        return False, "non-finite gradient of logp at the initial point"
    return True, ""


_MODEL_CACHE: Dict[tuple, Any] = {}


def load_pymc_model_cached(name: str, models_dir: Path):
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


def prior_predict_p_left(
    model_names: List[str],
    models_dir: Path,
    feature_row: Dict[str, Any],
    *,
    var_name: str = "p_left",
    n_samples: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """Prior-predictive mean of `p_left` for each model on a single stimulus.

    `feature_row` is a dict of feature-column → value. Must include every
    `pm.Data` input the model expects, including the observed-response container
    (whose value is unused for `p_left` predictions — pass a dummy 0/1).

    No MCMC — samples `p_left` from each model's prior under the given stimulus,
    averages over draws, returns one scalar per model.
    """
    pm = _import_pymc()
    out: Dict[str, float] = {}
    for name in model_names:
        model = load_pymc_model_cached(name, models_dir)
        stim_data = make_stim_data(model, [feature_row])
        with model:
            pm.set_data(stim_data)
            ppc = pm.sample_prior_predictive(
                draws=n_samples,
                var_names=[var_name],
                random_seed=seed,
            )
        arr = validate_probability_array(
            ppc.prior[var_name].values,
            context=f"Model {name!r} prior-predictive {var_name}",
        )  # shape: (chain, draw, n_stim=1)
        out[name] = float(arr.mean())
    return out


def prior_predict_p_left_draws(
    model_names: List[str],
    models_dir: Path,
    feature_rows: List[Dict[str, Any]],
    *,
    var_name: str = "p_left",
    n_samples: int = 200,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Per-draw prior-predictive `p_left` for each model over a *batch* of stimuli.

    Binds all ``feature_rows`` into the model's ``pm.Data`` containers at once
    and runs a single ``sample_prior_predictive`` per model, so the fixed
    per-call cost (graph compilation, sampling setup) is paid once per model
    instead of once per model *per stimulus*. Returns
    ``{model_name: array of shape (n_draws, n_rows)}`` — the full draws, which
    joint-EIG selection needs to see the correlation that shared parameters
    induce between stimuli within a model.

    With the same ``seed``, the prior parameter draws are identical to the
    per-row path's (priors do not depend on the data).
    """
    if not feature_rows:
        raise ValueError("feature_rows must be non-empty.")
    pm = _import_pymc()
    out: Dict[str, np.ndarray] = {}
    for name in model_names:
        model = load_pymc_model_cached(name, models_dir)
        stim_data = make_stim_data(model, feature_rows)
        with model:
            pm.set_data(stim_data)
            ppc = pm.sample_prior_predictive(
                draws=n_samples,
                var_names=[var_name],
                random_seed=seed,
            )
        arr = validate_probability_array(
            ppc.prior[var_name].values,
            context=f"Model {name!r} prior-predictive {var_name}",
        )  # shape: (chain, draw, n_rows)
        draws = arr.reshape(-1, arr.shape[-1])
        if draws.shape != (n_samples, len(feature_rows)):
            raise ValueError(
                f"Model {name!r}: batched {var_name} has shape {arr.shape}, "
                f"expected per-stimulus axis of length {len(feature_rows)} — "
                "is the model's p_left per-stimulus?"
            )
        out[name] = draws
    return out


def prior_predict_p_left_batch(
    model_names: List[str],
    models_dir: Path,
    feature_rows: List[Dict[str, Any]],
    *,
    var_name: str = "p_left",
    n_samples: int = 200,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Prior-predictive mean of `p_left` for each model over a *batch* of stimuli.

    Mean over the draws of :func:`prior_predict_p_left_draws`; returns
    ``{model_name: array of shape (n_rows,)}``. With the same ``seed`` the
    means match :func:`prior_predict_p_left` row for row.
    """
    draws = prior_predict_p_left_draws(
        model_names,
        models_dir,
        feature_rows,
        var_name=var_name,
        n_samples=n_samples,
        seed=seed,
    )
    return {name: arr.mean(axis=0) for name, arr in draws.items()}


def eig_from_prior_means(
    preds: Dict[str, float],
    model_weights: Optional[Dict[str, float]] = None,
) -> float:
    """EIG (bits) of one stimulus from per-model prior-predictive p_left means.

    Standard formula: EIG = H(M) - E_R[H(M|R)] for a binary response R, with
    the model prior taken from `model_weights` (uniform if omitted/degenerate).
    """
    import math

    if not preds:
        return 0.0
    preds = {
        name: validate_probability(value, context=f"prediction for model {name!r}")
        for name, value in preds.items()
    }
    if model_weights:
        model_weights = validate_theory_weights(model_weights)
        total_w = math.fsum(model_weights.get(m, 0.0) for m in preds)
        if total_w <= 0:
            p_model = {m: 1.0 / len(preds) for m in preds}
        else:
            p_model = {m: model_weights.get(m, 0.0) / total_w for m in preds}
    else:
        p_model = {m: 1.0 / len(preds) for m in preds}

    p_left = sum(preds[m] * p_model[m] for m in preds)
    p_right = 1.0 - p_left
    if p_left <= 0 or p_right <= 0:
        return 0.0

    def h_given_r(response_is_left: bool) -> float:
        denom = p_left if response_is_left else p_right
        p_m_r = []
        for m in preds:
            lik = preds[m] if response_is_left else (1.0 - preds[m])
            p_m_r.append(lik * p_model[m] / denom)
        return -sum(p * math.log2(p) for p in p_m_r if p > 0)

    h_m = -sum(p * math.log2(p) for p in p_model.values() if p > 0)
    h_m_given_r = p_left * h_given_r(True) + p_right * h_given_r(False)
    return max(0.0, h_m - h_m_given_r)


def expected_information_gain_prior_pymc(
    feature_row: Dict[str, Any],
    model_names: List[str],
    models_dir: Path,
    *,
    model_weights: Optional[Dict[str, float]] = None,
    n_samples: int = 200,
    seed: int = 42,
) -> float:
    """EIG of a candidate stimulus computed from prior-predictive p_left per model.

    Standard formula: EIG = H(M) - E_R[H(M|R)] in bits.
    `feature_row` must include every pm.Data input of the models (including a
    dummy observed-response value, which is ignored for p_left).
    """
    preds = prior_predict_p_left(
        model_names,
        models_dir,
        feature_row,
        n_samples=n_samples,
        seed=seed,
    )
    return eig_from_prior_means(preds, model_weights)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# Default MCMC sampler settings for ``fit_model``, kept as a single source of
# truth so the cache key can fold the *resolved* settings in. A fit's posterior
# depends on draws/tune/chains/cores/seed/target_accept, so a cache keyed only on
# (model, data) would silently reuse a posterior sampled under different settings
# if a cache_dir is shared across callers that request different settings (e.g. a
# standalone CLI run pointed at the inner loop's cache_dir). draws/tune/chains/
# target_accept derive from src.models.mcmc_defaults so the production values live
# in exactly one place (no drift with the CLI defaults).
_FIT_DEFAULTS = {
    "draws": PRODUCTION_DRAWS,
    "tune": PRODUCTION_TUNE,
    "chains": PRODUCTION_CHAINS,
    "cores": PRODUCTION_CORES,
    "random_seed": 42,
    "target_accept": PRODUCTION_TARGET_ACCEPT,
    # NUTS trajectory-length cap. 10 is PyMC's default (effectively uncapped). A
    # caller can lower it to bound per-iteration work so a pathologically stiff
    # model (weak identifiability -> the sampler wants ever-deeper trees) can't
    # hang a fit; the model still fits and competes, just with bounded cost.
    "max_treedepth": 10,
}

# Name of the optional module-level dict a model `.py` may declare to request
# sampler settings suited to *its own* posterior geometry.
SAMPLER_SETTINGS_NAME = "SAMPLER_SETTINGS"

# Cache of validated per-model declarations, keyed by (path, content hash) so a
# rewritten model file is always re-read (the inner loop rewrites candidates).
_SAMPLER_SETTINGS_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _validated_sampler_settings(declared: Any, source: Path) -> Dict[str, Any]:
    """Check a model's ``SAMPLER_SETTINGS`` declaration and return it as a dict.

    Fails loudly rather than ignoring anything it does not understand: a typo'd
    key (``targt_accept``) or a non-numeric value would otherwise silently leave
    the model sampling under the global defaults, which is exactly the kind of
    quiet mis-configuration this project forbids.
    """
    if not isinstance(declared, dict):
        raise TypeError(
            f"{source}: {SAMPLER_SETTINGS_NAME} must be a dict of sampler setting "
            f"-> number, got {type(declared).__name__}."
        )
    for key, value in declared.items():
        if key not in _FIT_DEFAULTS:
            raise ValueError(
                f"{source}: {SAMPLER_SETTINGS_NAME} declares unknown sampler "
                f"setting {key!r}. Valid settings: {sorted(_FIT_DEFAULTS)}."
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{source}: {SAMPLER_SETTINGS_NAME}[{key!r}] must be a number, "
                f"got {type(value).__name__} ({value!r})."
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"{source}: {SAMPLER_SETTINGS_NAME}[{key!r}] is not finite ({value!r})."
            )
    return dict(declared)


def model_sampler_settings(name: str, models_dir: Path) -> Dict[str, Any]:
    """The validated sampler settings the model file declares (``{}`` if none).

    Read by importing the file directly rather than through
    :func:`load_pymc_model`, because the cache key must be computable for any
    file the fitter will be handed — including one that builds no ``pm.Model``.
    """
    py_path = Path(models_dir) / f"{name}.py"
    key = (str(py_path.resolve()), _sha256_file(py_path))
    if key not in _SAMPLER_SETTINGS_CACHE:
        mod = _exec_model_module(py_path, mod_prefix="_pymc_sampler_settings_")
        _SAMPLER_SETTINGS_CACHE[key] = _validated_sampler_settings(
            getattr(mod, SAMPLER_SETTINGS_NAME, {}), py_path
        )
    return dict(_SAMPLER_SETTINGS_CACHE[key])


def resolve_fit_settings(
    name: str, models_dir: Path, explicit: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Resolve the sampler settings for one fit. THE single resolution point.

    Precedence, highest first:

    1. an explicit caller value (anything in ``explicit`` that is not ``None``);
    2. the model file's own ``SAMPLER_SETTINGS`` declaration;
    3. :data:`_FIT_DEFAULTS` (the centralized production config).

    ``None`` in ``explicit`` means "the caller did not ask for anything", which
    is why ``fit_model``'s sampler arguments default to ``None`` instead of to
    the production values — a pinned default is indistinguishable from a
    deliberate request and would silently outrank the model's declaration.

    Both cache keys (the on-disk ``.nc`` fingerprint and the in-process
    ``_cache_key``) are built from the dict this returns, so they cannot
    disagree about which settings a stored fit was sampled under.
    """
    given = {k: v for k, v in (explicit or {}).items() if v is not None}
    unknown = sorted(set(given) - set(_FIT_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown sampler setting(s) {unknown} requested for model {name!r}. "
            f"Valid settings: {sorted(_FIT_DEFAULTS)}."
        )
    return {**_FIT_DEFAULTS, **model_sampler_settings(name, models_dir), **given}


def _sampler_signature(fit_kwargs: Dict[str, Any]) -> str:
    """Stable string of the resolved sampler settings, for cache keying."""
    merged = {**_FIT_DEFAULTS, **fit_kwargs}
    return ";".join(f"{k}={merged[k]}" for k in sorted(_FIT_DEFAULTS))


def _thin_posterior(idata: Any, max_draws: int) -> Any:
    """Subsample an InferenceData's posterior to at most ``max_draws`` samples.

    Keeps ``max_draws // n_chains`` draws of each chain by an even stride across
    the whole chain (deterministic), so the thinned posterior spans the full chain
    rather than only its earliest, least-mixed draws. A downstream
    posterior-predictive pass over many stimuli then builds a far smaller
    ``(chain, draw, n_stim)`` array. Returns the idata unchanged when it already
    holds ``<= max_draws`` total samples.
    """
    n_chains = int(idata.posterior.sizes["chain"])
    n_draws = int(idata.posterior.sizes["draw"])
    if n_chains * n_draws <= max_draws:
        return idata
    per_chain = max(1, max_draws // n_chains)
    idx = np.linspace(0, n_draws - 1, num=per_chain, dtype=int)
    return idata.isel(draw=idx)


@dataclass
class FittedModel:
    """A fitted PyMC model and its InferenceData."""

    name: str
    model: Any  # pm.Model
    idata: Any  # az.InferenceData
    fingerprint: str

    def predict_p_left_draws(
        self,
        stim_data: Dict[str, np.ndarray],
        *,
        var_name: str = "p_left",
        seed: int = 42,
        max_draws: Optional[int] = None,
    ) -> np.ndarray:
        """Per-draw posterior-predictive p_left for each stimulus row.

        `stim_data` must include every pm.Data input expected by the model
        (the observed-response container can be set to dummies — it is unused).
        Returns shape (n_draws, n_stim) with chains flattened — the posterior
        counterpart of ``prior_predict_p_left_draws``, e.g. for joint-EIG
        stimulus selection under a fitted model.

        ``max_draws`` thins the posterior to at most that many samples before
        the posterior-predictive pass. The intermediate array scales with
        draws × n_stim, so thinning keeps memory bounded when predicting over
        very large stimulus sets (e.g. an exhaustive design pool).
        """
        pm = _import_pymc()
        idata = self.idata if max_draws is None else _thin_posterior(self.idata, max_draws)
        with self.model:
            pm.set_data(stim_data)
            pp = pm.sample_posterior_predictive(
                idata,
                var_names=[var_name],
                random_seed=seed,
                progressbar=False,
            )
        arr = validate_probability_array(
            pp.posterior_predictive[var_name].values,
            context=f"Model {self.name!r} posterior-predictive {var_name}",
        )  # (chain, draw, n_stim)
        draws = arr.reshape(-1, arr.shape[-1])
        # Count trials from the observed-response container, which is per-trial by
        # construction, rather than from an arbitrary first entry of ``stim_data``.
        # Not every container is trial-aligned: a model using a ``prepare_observed``
        # hook may bind a unique-sequence table whose length is unrelated to the
        # number of stimuli, and dict order would decide whether this check passed.
        n_stim = len(stim_data[observed_response_data(self.model)])
        if draws.shape[1] != n_stim:
            raise ValueError(
                f"Model {self.name!r}: posterior-predictive {var_name} has shape "
                f"{arr.shape}, expected per-stimulus axis of length {n_stim} — "
                "is the model's p_left per-stimulus?"
            )
        return draws

    def predict_p_left(
        self,
        stim_data: Dict[str, np.ndarray],
        *,
        var_name: str = "p_left",
        seed: int = 42,
        max_draws: Optional[int] = None,
    ) -> np.ndarray:
        """Posterior-mean p_left for each stimulus row in `stim_data`.

        Mean over the draws of :meth:`predict_p_left_draws`; returns shape
        (n_stim,). See that method for the `stim_data` and `max_draws` contract.
        """
        return self.predict_p_left_draws(
            stim_data, var_name=var_name, seed=seed, max_draws=max_draws
        ).mean(axis=0)

    def elpd_loo(self) -> float:
        """Expected log pointwise predictive density (PSIS-LOO).

        PSIS-LOO is only trustworthy when the importance-sampling Pareto-k tail
        index stays low; ArviZ sets ``loo.warning`` when too many points exceed
        the safe threshold. We do not silently return a number ArviZ flagged as
        unreliable — surface an attributed warning so a dubious score is visible
        in the run log (the value is still returned; the human/comparison can act
        on the warning).
        """
        az = _import_arviz()
        loo = az.loo(self.idata)
        if getattr(loo, "warning", False):
            print(
                f"  [warn] {self.name}: PSIS-LOO is unreliable (many high Pareto-k "
                "points); its ELPD-LOO may be inaccurate.",
                file=sys.stderr,
                flush=True,
            )
        return float(loo.elpd_loo)

    def sample_synthetic_responses(
        self, stim_data: Dict[str, np.ndarray], *, n_datasets: int, seed: int = 42
    ) -> np.ndarray:
        """Posterior-predictive samples of the observed response.

        Returns array shape (n_datasets, n_stim) of integer responses, one
        synthetic dataset per row. Caps n_datasets at chains*draws of the
        stored idata; raises if asked for more.
        """
        pm = _import_pymc()
        n_chains = int(self.idata.posterior.sizes["chain"])
        n_draws = int(self.idata.posterior.sizes["draw"])
        capacity = n_chains * n_draws
        if n_datasets > capacity:
            raise ValueError(
                f"Requested {n_datasets} synthetic datasets but posterior only has "
                f"{n_chains} chains × {n_draws} draws = {capacity}. Increase chains/draws or reduce n_datasets."
            )

        response_rv_name = self.model.observed_RVs[0].name
        with self.model:
            pm.set_data(stim_data)
            pp = pm.sample_posterior_predictive(
                self.idata,
                var_names=[response_rv_name],
                random_seed=seed,
                progressbar=False,
            )
        arr = pp.posterior_predictive[response_rv_name].values  # (chain, draw, n_stim)
        flat = arr.reshape(-1, arr.shape[-1])  # (chain*draw, n_stim)
        if n_datasets >= flat.shape[0]:
            return flat
        # Subsample WITHOUT replacement across the full chain×draw pool rather than
        # taking flat[:n_datasets] — the reshape above is chain-major, so a head
        # slice would draw the PPC null distribution from a single chain's first
        # draws (autocorrelated, ignoring the other chains). A seeded, evenly
        # strided selection spreads the replicates across all chains/draws and is
        # reproducible for a given seed.
        idx = np.linspace(0, flat.shape[0] - 1, num=n_datasets, dtype=int)
        return flat[idx]


_FIT_CACHE: Dict[tuple, FittedModel] = {}


def _cache_key(
    name: str,
    models_dir: Path,
    csv_path: Path,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple:
    return (
        name,
        _sha256_file(models_dir / f"{name}.py"),
        _sha256_file(csv_path),
        _sampler_signature(resolve_fit_settings(name, models_dir, fit_kwargs)),
    )


def fit_model(
    name: str,
    models_dir: Path,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    draws: Optional[int] = None,
    tune: Optional[int] = None,
    chains: Optional[int] = None,
    cores: Optional[int] = None,
    random_seed: Optional[int] = None,
    target_accept: Optional[float] = None,
    max_treedepth: Optional[int] = None,
) -> FittedModel:
    """Load the named PyMC model, fit it on `responses_path`, return a FittedModel.

    Every sampler argument defaults to ``None``, meaning "unset — resolve it".
    :func:`resolve_fit_settings` then applies an explicit caller value first, the
    model file's own ``SAMPLER_SETTINGS`` declaration next, and the centralized
    production defaults last. Defaulting these to the production values instead
    would make "the caller wants 0.99" indistinguishable from "the caller said
    nothing", silently overriding every model-declared setting.

    If `cache_dir` is given and `<cache_dir>/<name>.<fingerprint>.nc` exists,
    load idata from disk instead of refitting.
    """
    pm = _import_pymc()
    az = _import_arviz()

    models_dir = Path(models_dir)
    responses_path = Path(responses_path)
    model = load_pymc_model(name, models_dir)

    settings = resolve_fit_settings(
        name,
        models_dir,
        {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "cores": cores,
            "random_seed": random_seed,
            "target_accept": target_accept,
            "max_treedepth": max_treedepth,
        },
    )

    # Fingerprint from the model source + the responses-file bytes + the resolved
    # sampler settings — the SAME inputs as the in-process ``_cache_key``, which
    # resolves through the same ``resolve_fit_settings``. Keeping the two keyed
    # identically means the on-disk ``.nc`` and the in-process cache can never
    # disagree about which fit corresponds to a (model, data, sampler) triple, so
    # the seeded critique always reuses exactly the fit the model comparison
    # scored, and a fit sampled under different draws/chains is never silently
    # reused for a request that asked for different settings.
    fp = hashlib.sha256(
        (
            _sha256_file(models_dir / f"{name}.py")
            + _sha256_file(responses_path)
            + _sampler_signature(settings)
        ).encode("utf-8")
    ).hexdigest()[:16]

    nc_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        nc_path = cache_dir / f"{name}.{fp}.nc"

    if nc_path is not None and nc_path.exists():
        idata = az.from_netcdf(str(nc_path))
        _warn_sampling_diagnostics(name, idata)
        return FittedModel(name=name, model=model, idata=idata, fingerprint=fp)

    observed = extract_observed(responses_path, model)
    with model:
        pm.set_data(observed)
        idata = pm.sample(
            draws=settings["draws"],
            tune=settings["tune"],
            chains=settings["chains"],
            cores=settings["cores"],
            target_accept=settings["target_accept"],
            max_treedepth=settings["max_treedepth"],
            progressbar=False,
            random_seed=settings["random_seed"],
            idata_kwargs={"log_likelihood": True},
        )

    _warn_sampling_diagnostics(name, idata)

    if nc_path is not None:
        idata.to_netcdf(str(nc_path))

    return FittedModel(name=name, model=model, idata=idata, fingerprint=fp)


def _divergence_count(idata: Any) -> Optional[int]:
    """Number of divergent transitions, or None if the trace does not record any.

    None is a real answer, not a failure: a sampler that is not NUTS (a model
    with discrete parameters falls back to Metropolis) writes no ``diverging``
    stat. It is deliberately NOT folded into 0 — "no divergences" and "nobody
    checked" must not look the same. Anything else raises.
    """
    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is None or "diverging" not in sample_stats:
        return None
    return int(sample_stats["diverging"].values.sum())


def _max_rhat(idata: Any) -> float:
    """Largest R-hat across variables; NaN when ArviZ cannot compute one.

    ArviZ returns NaN (not an error) for a single-chain trace, where R-hat is
    undefined. That NaN is reported as "unverified", never as "converged".
    """
    az = _import_arviz()
    rhat = az.rhat(idata)
    return max((float(rhat[v].max()) for v in rhat.data_vars), default=float("nan"))


def _warn_sampling_diagnostics(name: str, idata: Any) -> None:
    """Loudly surface NUTS trouble (divergences, poor R-hat) for a fit.

    These are advisory, not fatal — ArviZ still returns usable arrays — but a fit
    with divergences or R-hat > 1.01 is suspect, and accepting its ELPD at face
    value is exactly the silent-quality trap the project's fail-loud rule guards
    against. Print an attributed warning so a degraded fit is visible in the run
    log. Diagnostics are rerun on cache hits so loading a suspect stored fit
    cannot make its warning disappear from a later run.

    A diagnostic that could not be computed is itself warned about: previously a
    missing ``diverging`` stat read as 0 divergences and an unavailable R-hat as
    NaN, i.e. the two values that mean "this fit is healthy".
    """
    n_div = _divergence_count(idata)
    if n_div is None:
        print(
            f"  [warn] {name}: the trace records no divergence statistic, so "
            "sampling quality could NOT be checked (did the sampler fall back "
            "off NUTS?).",
            file=sys.stderr,
            flush=True,
        )
    elif n_div > 0:
        print(
            f"  [warn] {name}: {n_div} divergence(s) during sampling; the posterior "
            "may be biased — treat its ELPD-LOO with caution.",
            file=sys.stderr,
            flush=True,
        )
    max_rhat = _max_rhat(idata)
    if not math.isfinite(max_rhat):
        print(
            f"  [warn] {name}: R-hat is unavailable (got {max_rhat}); convergence "
            "was NOT verified — a single-chain fit cannot report one.",
            file=sys.stderr,
            flush=True,
        )
    elif max_rhat > 1.01:
        print(
            f"  [warn] {name}: max R-hat={max_rhat:.3f} (>1.01); chains may not have "
            "converged.",
            file=sys.stderr,
            flush=True,
        )


def fit_models_cached(
    model_names: List[str],
    models_dir: Path,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    **fit_kwargs: Any,
) -> Dict[str, FittedModel]:
    """Fit each model in `model_names`, reusing cached fits keyed by
    (model_name, sha256(model.py), sha256(responses.csv), sampler settings). Each
    call to `pm.sample` is expensive, so identical (model, data, sampler) triples
    are reused within a process. If `cache_dir` is given, also persists/reads .nc
    files (keyed by the same triple).
    """
    models_dir = Path(models_dir)
    responses_path = Path(responses_path)
    out: Dict[str, FittedModel] = {}
    for name in model_names:
        key = _cache_key(name, models_dir, responses_path, fit_kwargs)
        cached = _FIT_CACHE.get(key)
        if cached is not None:
            _warn_sampling_diagnostics(name, cached.idata)
            out[name] = cached
            continue
        fitted = fit_model(
            name, models_dir, responses_path, cache_dir=cache_dir, **fit_kwargs
        )
        _FIT_CACHE[key] = fitted
        out[name] = fitted
    return out


def clear_fit_cache() -> None:
    """Clear the in-process fit cache. Useful for tests."""
    _FIT_CACHE.clear()


def evict_fit_cache(model_name: str) -> int:
    """Drop every cached fit for ``model_name``; return how many were evicted.

    Used when the inner loop prunes a losing model — its InferenceData would
    otherwise stay resident in the in-process cache for the rest of the run.
    The cache key leads with the model name (see ``_cache_key``).
    """
    keys = [k for k in _FIT_CACHE if k and k[0] == model_name]
    for k in keys:
        del _FIT_CACHE[k]
    return len(keys)
