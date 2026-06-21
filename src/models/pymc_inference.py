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


# Attribute under which a loaded model carries its optional theorist-supplied
# featurizer (a ``compute_features(sequence_a, sequence_b) -> dict`` callable).
_EXTRA_FEATURIZER_ATTR = "_auto_psych_extra_featurizer"


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


def make_stim_data(model, rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Build a `pm.set_data` dict from a list of row dicts for a given model.

    Each `pm.Data` container in `model` is filled with the corresponding column
    from `rows`, cast to the placeholder's dtype. Useful for predict_p_left and
    sample_synthetic_responses, where the caller has rows but not a CSV file.

    If the model declares a ``compute_features`` featurizer, its extra columns
    are computed from each row's raw sequences first.
    """
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
    """
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
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
    """
    pm = _import_pymc()
    model = load_pymc_model(name, models_dir)
    try:
        observed = extract_observed(responses_path, model)
        with model:
            pm.set_data(observed)
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
    except Exception as e:  # a graph that cannot even be evaluated
        return False, f"logp evaluation raised: {type(e).__name__}: {e}"
    if not math.isfinite(logp):
        return False, f"non-finite logp ({logp}) at the initial point"
    try:
        grad = np.asarray(model.compile_dlogp()(point), dtype=float)
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


# Default MCMC sampler settings for ``fit_model``, kept as a single source of
# truth so the cache key can fold the *resolved* settings in. A fit's posterior
# depends on draws/tune/chains/cores/seed, so a cache keyed only on (model, data)
# would silently reuse a posterior sampled under different settings if a cache_dir
# is shared across callers that request different settings (e.g. a standalone CLI
# run pointed at the inner loop's cache_dir).
_FIT_DEFAULTS = {"draws": 2000, "tune": 2000, "chains": 4, "cores": 1, "random_seed": 42}


def _sampler_signature(fit_kwargs: Dict[str, Any]) -> str:
    """Stable string of the resolved sampler settings, for cache keying."""
    merged = {**_FIT_DEFAULTS, **fit_kwargs}
    return ";".join(f"{k}={merged[k]}" for k in sorted(_FIT_DEFAULTS))


def _sha256_dict_arrays(d: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(d.keys()):
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
        h.update(np.ascontiguousarray(d[k]).tobytes())
        h.update(np.array(d[k].shape, dtype="int64").tobytes())
        h.update(d[k].dtype.str.encode("utf-8"))
    return h.hexdigest()


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

    def predict_p_left(
        self,
        stim_data: Dict[str, np.ndarray],
        *,
        var_name: str = "p_left",
        seed: int = 42,
        max_draws: Optional[int] = None,
    ) -> np.ndarray:
        """Posterior-mean p_left for each stimulus row in `stim_data`.

        `stim_data` must include every pm.Data input expected by the model
        (the observed-response container can be set to dummies — it is unused).
        Returns shape (n_stim,).

        ``max_draws`` thins the posterior to at most that many samples before the
        posterior-predictive pass. The intermediate ``(chain, draw, n_stim)``
        array scales with draws × n_stim, so thinning keeps memory bounded when
        predicting over very large stimulus sets (e.g. an exhaustive eval pool);
        the posterior *mean* is essentially unchanged by using fewer samples.
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
        arr = pp.posterior_predictive[var_name]
        return arr.mean(("chain", "draw")).values

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
        _sampler_signature(fit_kwargs or {}),
    )


def fit_model(
    name: str,
    models_dir: Path,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    draws: int = _FIT_DEFAULTS["draws"],
    tune: int = _FIT_DEFAULTS["tune"],
    chains: int = _FIT_DEFAULTS["chains"],
    cores: int = _FIT_DEFAULTS["cores"],
    random_seed: int = _FIT_DEFAULTS["random_seed"],
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

    # Fingerprint from the model source + the responses-file bytes + the resolved
    # sampler settings — the SAME inputs as the in-process ``_cache_key``. Keeping
    # the two keyed identically means the on-disk ``.nc`` and the in-process cache
    # can never disagree about which fit corresponds to a (model, data, sampler)
    # triple, so the seeded critique always reuses exactly the fit the model
    # comparison scored, and a fit sampled under different draws/chains is never
    # silently reused for a request that asked for different settings.
    fp = hashlib.sha256(
        (
            _sha256_file(models_dir / f"{name}.py")
            + _sha256_file(responses_path)
            + _sampler_signature(
                {
                    "draws": draws,
                    "tune": tune,
                    "chains": chains,
                    "cores": cores,
                    "random_seed": random_seed,
                }
            )
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

    observed = extract_observed(responses_path, model)
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

    _warn_sampling_diagnostics(name, idata)

    if nc_path is not None:
        idata.to_netcdf(str(nc_path))

    return FittedModel(name=name, model=model, idata=idata, fingerprint=fp)


def _warn_sampling_diagnostics(name: str, idata: Any) -> None:
    """Loudly surface NUTS trouble (divergences, poor R-hat) for a fresh fit.

    These are advisory, not fatal — ArviZ still returns usable arrays — but a fit
    with divergences or R-hat > 1.01 is suspect, and accepting its ELPD at face
    value is exactly the silent-quality trap the project's fail-loud rule guards
    against. Print an attributed warning so a degraded fit is visible in the run
    log. Only called on a real sample (not on a cache hit).
    """
    az = _import_arviz()
    try:
        n_div = int(idata.sample_stats["diverging"].values.sum())
    except Exception:
        n_div = 0
    if n_div > 0:
        print(
            f"  [warn] {name}: {n_div} divergence(s) during sampling; the posterior "
            "may be biased — treat its ELPD-LOO with caution.",
            file=sys.stderr,
            flush=True,
        )
    try:
        rhat = az.rhat(idata)
        max_rhat = max(float(rhat[v].max()) for v in rhat.data_vars)
    except Exception:
        max_rhat = float("nan")
    if math.isfinite(max_rhat) and max_rhat > 1.01:
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
