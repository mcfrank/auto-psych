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

# Imports of pymc / arviz / pytensor are local in each function so that
# loading this module is cheap when only e.g. cache utilities are used.


def _import_pymc():
    import pymc as pm

    return pm


def _import_arviz():
    import arviz as az

    return az


def load_pymc_model(name: str, models_dir: Path):
    """Import `models_dir/<name>.py` and return its module-level `model` attribute.

    Fails loudly if the file is missing, fails to import, or does not expose a
    `pm.Model` at module level.
    """
    pm = _import_pymc()
    models_dir = Path(models_dir)
    py_path = models_dir / f"{name}.py"
    if not py_path.exists():
        raise FileNotFoundError(f"PyMC model file not found: {py_path}")

    unique_mod_name = (
        f"_pymc_model_{name}_{hashlib.sha1(str(py_path).encode()).hexdigest()[:8]}"
    )
    spec = importlib.util.spec_from_file_location(
        unique_mod_name, py_path, submodule_search_locations=[]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build module spec for {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_mod_name] = mod
    spec.loader.exec_module(mod)

    model = getattr(mod, "model", None)
    if not isinstance(model, pm.Model):
        raise TypeError(
            f"{py_path} must define a module-level `model: pm.Model` "
            f"(got {type(model).__name__ if model is not None else 'missing'})"
        )
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


def make_stim_data(model, rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Build a `pm.set_data` dict from a list of row dicts for a given model.

    Each `pm.Data` container in `model` is filled with the corresponding column
    from `rows`, cast to the placeholder's dtype. Useful for predict_p_left and
    sample_synthetic_responses, where the caller has rows but not a CSV file.
    """
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
    """
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

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
    """
    pm = _import_pymc()
    model = load_pymc_model(name, models_dir)
    observed = extract_observed(responses_path, model)
    with model:
        pm.set_data(observed)
    try:
        logp = float(model.compile_logp()(model.initial_point()))
    except Exception as e:  # a graph that cannot even be evaluated
        return False, f"logp evaluation raised: {type(e).__name__}: {e}"
    if not math.isfinite(logp):
        return False, f"non-finite logp ({logp}) at the initial point"
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
        arr = ppc.prior[var_name].values  # shape: (chain, draw, n_stim=1)
        out[name] = float(arr.mean())
    return out


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
    import math

    preds = prior_predict_p_left(
        model_names,
        models_dir,
        feature_row,
        n_samples=n_samples,
        seed=seed,
    )
    if not preds:
        return 0.0
    if model_weights:
        total_w = sum(model_weights.get(m, 0.0) for m in preds)
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _sha256_dict_arrays(d: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(d.keys()):
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
        h.update(np.ascontiguousarray(d[k]).tobytes())
        h.update(np.array(d[k].shape, dtype="int64").tobytes())
        h.update(d[k].dtype.str.encode("utf-8"))
    return h.hexdigest()


@dataclass
class FittedModel:
    """A fitted PyMC model and its InferenceData."""

    name: str
    model: Any  # pm.Model
    idata: Any  # az.InferenceData
    fingerprint: str

    def predict_p_left(
        self,
        stim_data: Dict[str, np.ndarray],
        *,
        var_name: str = "p_left",
        seed: int = 42,
    ) -> np.ndarray:
        """Posterior-mean p_left for each stimulus row in `stim_data`.

        `stim_data` must include every pm.Data input expected by the model
        (the observed-response container can be set to dummies — it is unused).
        Returns shape (n_stim,).
        """
        pm = _import_pymc()
        with self.model:
            pm.set_data(stim_data)
            pp = pm.sample_posterior_predictive(
                self.idata,
                var_names=[var_name],
                random_seed=seed,
                progressbar=False,
            )
        arr = pp.posterior_predictive[var_name]
        return arr.mean(("chain", "draw")).values

    def elpd_loo(self) -> float:
        """Expected log pointwise predictive density (PSIS-LOO)."""
        az = _import_arviz()
        loo = az.loo(self.idata)
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
        return flat[:n_datasets]


_FIT_CACHE: Dict[tuple, FittedModel] = {}


def _cache_key(name: str, models_dir: Path, csv_path: Path) -> tuple:
    return (name, _sha256_file(models_dir / f"{name}.py"), _sha256_file(csv_path))


def fit_model(
    name: str,
    models_dir: Path,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    cores: int = 1,
    random_seed: int = 42,
) -> FittedModel:
    """Load the named PyMC model, fit it on `responses_path`, return a FittedModel.

    If `cache_dir` is given and `<cache_dir>/<name>.<fingerprint>.nc` exists,
    load idata from disk instead of refitting.
    """
    pm = _import_pymc()
    az = _import_arviz()

    models_dir = Path(models_dir)
    responses_path = Path(responses_path)
    model = load_pymc_model(name, models_dir)

    observed = extract_observed(responses_path, model)
    fp = hashlib.sha256(
        (
            _sha256_file(models_dir / f"{name}.py") + _sha256_dict_arrays(observed)
        ).encode("utf-8")
    ).hexdigest()[:16]

    nc_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        nc_path = cache_dir / f"{name}.{fp}.nc"

    if nc_path is not None and nc_path.exists():
        idata = az.from_netcdf(str(nc_path))
        return FittedModel(name=name, model=model, idata=idata, fingerprint=fp)

    with model:
        pm.set_data(observed)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            progressbar=False,
            random_seed=random_seed,
            idata_kwargs={"log_likelihood": True},
        )

    if nc_path is not None:
        idata.to_netcdf(str(nc_path))

    return FittedModel(name=name, model=model, idata=idata, fingerprint=fp)


def fit_models_cached(
    model_names: List[str],
    models_dir: Path,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    **fit_kwargs: Any,
) -> Dict[str, FittedModel]:
    """Fit each model in `model_names`, reusing cached fits keyed by
    (model_name, sha256(model.py), sha256(responses.csv)). Each call to
    `pm.sample` is expensive, so identical (model, data) pairs are reused
    within a process. If `cache_dir` is given, also persists/reads .nc files.
    """
    models_dir = Path(models_dir)
    responses_path = Path(responses_path)
    out: Dict[str, FittedModel] = {}
    for name in model_names:
        key = _cache_key(name, models_dir, responses_path)
        cached = _FIT_CACHE.get(key)
        if cached is not None:
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
