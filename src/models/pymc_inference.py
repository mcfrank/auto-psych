"""PyMC inference bridge for cognitive models.

Each model is a ``<name>.py`` file with a module-level
``with pm.Model() as model:`` block. This module loads those models, fits them
to observed data via MCMC, and exposes posterior-mean predictions, ELPD-LOO
for Bayesian model comparison, and posterior-predictive samples for PPC.

Convention: every ``pm.Data`` container in the model must have a name matching
a column in the preprocessed responses CSV. The bridge auto-pulls
``df[name].values`` for each container. The observed-response container is
identified by tracing ``model.observed_RVs[0]`` back through the pytensor graph
to its ``TensorSharedVariable`` ancestor.

Model loading and data binding live in ``model_loading`` and ``data_binding``;
this module imports the names it needs from those children and adds prediction,
fitting, and diagnostics.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import sys
import tempfile
import traceback
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.models.mcmc_defaults import (
    ESCALATED_TARGET_ACCEPT,
    MAX_DIVERGENCE_FRACTION,
    MAX_R_HAT,
    MIN_BULK_ESS,
    PRODUCTION_CHAINS,
    PRODUCTION_CORES,
    PRODUCTION_DRAWS,
    PRODUCTION_TARGET_ACCEPT,
    PRODUCTION_TUNE,
)
from src.models.loo_reliability import (
    LooDiagnostics,
    describe_unreliable,
    loo_diagnostics,
)
from src.models.probability import validate_probability, validate_probability_array
from src.registry.io import validate_theory_weights

from src.models.model_loading import (
    _exec_model_module,
    _import_arviz,
    _import_pymc,
    load_pymc_model,
    load_pymc_model_cached,
    observed_response_data,
)
from src.models.data_binding import (
    extract_observed,
    make_stim_data,
)


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


# ---------------------------------------------------------------------------
# Prior prediction and EIG
# ---------------------------------------------------------------------------

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

    `feature_row` is a dict of feature-column -> value. Must include every
    `pm.Data` input the model expects, including the observed-response container
    (whose value is unused for `p_left` predictions -- pass a dummy 0/1).

    No MCMC -- samples `p_left` from each model's prior under the given stimulus,
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
    ``{model_name: array of shape (n_draws, n_rows)}`` -- the full draws, which
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
                f"expected per-stimulus axis of length {len(feature_rows)} -- "
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


# ---------------------------------------------------------------------------
# Fitting infrastructure
# ---------------------------------------------------------------------------

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
    file the fitter will be handed -- including one that builds no ``pm.Model``.
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

    1. an explicit caller value (anything in ``explicit`` that is not ``None``),
       except that a declared ``target_accept`` is a floor on the caller's;
    2. the model file's own ``SAMPLER_SETTINGS`` declaration;
    3. :data:`_FIT_DEFAULTS` (the centralized production config).

    ``None`` in ``explicit`` means "the caller did not ask for anything", which
    is why ``fit_model``'s sampler arguments default to ``None`` instead of to
    the production values -- a pinned default is indistinguishable from a
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
    declared = model_sampler_settings(name, models_dir)
    settings = {**_FIT_DEFAULTS, **declared, **given}
    # A model's declared target_accept is a floor: the loop passes one
    # explicitly, which used to override a model that needs smaller steps
    # (a model declaring 0.9 ran at the sweep's 0.8), and declaring a higher value
    # is how a candidate rejected for divergences fixes itself.
    if "target_accept" in declared:
        settings["target_accept"] = max(settings["target_accept"], declared["target_accept"])
    return settings


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
    # PSIS-LOO is computed once per fit (it is a full importance-sampling pass
    # over draws x trials) and shared by elpd_loo() and compare_table().
    _loo_diagnostics: Optional[LooDiagnostics] = field(
        default=None, init=False, repr=False, compare=False
    )

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
        (the observed-response container can be set to dummies -- it is unused).
        Returns shape (n_draws, n_stim) with chains flattened -- the posterior
        counterpart of ``prior_predict_p_left_draws``, e.g. for joint-EIG
        stimulus selection under a fitted model.

        ``max_draws`` thins the posterior to at most that many samples before
        the posterior-predictive pass. The intermediate array scales with
        draws x n_stim, so thinning keeps memory bounded when predicting over
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
                f"{arr.shape}, expected per-stimulus axis of length {n_stim} -- "
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

    def loo_diagnostics(self) -> LooDiagnostics:
        """PSIS-LOO of this fit with its reliability verdict, computed once.

        See ``src.models.loo_reliability``: trials whose log-likelihood is
        constant across draws are exact (not "unreliable", whatever arviz's
        blanket flag says), and the verdict rests on the proportion of the
        remaining trials with a high Pareto k.
        """
        if self._loo_diagnostics is None:
            self._loo_diagnostics = loo_diagnostics(self.idata)
        return self._loo_diagnostics

    def convergence_problems(self) -> List[str]:
        """Why this fit has not converged (empty when it has); see
        :func:`convergence_problems`."""
        return convergence_problems_of(self)

    def elpd_loo(self) -> float:
        """Expected log pointwise predictive density (PSIS-LOO).

        We do not silently return a number the diagnostic judged unreliable --
        an attributed, number-bearing warning goes to the run log (the value is
        still returned; the comparison acts on the same verdict through
        ``compare_table``'s ``loo_unreliable``).
        """
        diag = self.loo_diagnostics()
        if diag.unreliable:
            print(
                f"  [warn] {describe_unreliable(self.name, diag)}",
                file=sys.stderr,
                flush=True,
            )
        return diag.elpd_loo

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
                f"{n_chains} chains x {n_draws} draws = {capacity}. Increase chains/draws or reduce n_datasets."
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
        # Subsample WITHOUT replacement across the full chain x draw pool rather than
        # taking flat[:n_datasets] -- the reshape above is chain-major, so a head
        # slice would draw the PPC null distribution from a single chain's first
        # draws (autocorrelated, ignoring the other chains). A seeded, evenly
        # strided selection spreads the replicates across all chains/draws and is
        # reproducible for a given seed.
        idx = np.linspace(0, flat.shape[0] - 1, num=n_datasets, dtype=int)
        return flat[idx]


# ---------------------------------------------------------------------------
# Fit caching
# ---------------------------------------------------------------------------

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


def fit_fingerprint(
    name: str, models_dir: Path, responses_path: Path, settings: Dict[str, Any]
) -> str:
    """The on-disk cache fingerprint of one fit.

    Built from the model source, the responses-file bytes and the *resolved*
    sampler settings -- the SAME inputs as the in-process ``_cache_key``, which
    resolves through the same ``resolve_fit_settings``. Keeping the two keyed
    identically means the on-disk ``.nc`` and the in-process cache can never
    disagree about which fit corresponds to a (model, data, sampler) triple, so
    the seeded critique always reuses exactly the fit the model comparison
    scored, and a fit sampled under different draws/chains is never silently
    reused for a request that asked for different settings. Public so an
    offline reader of a finished run's cache (the LOO design-effect analysis)
    can locate the fit the loop scored without re-deriving the formula.
    """
    return hashlib.sha256(
        (
            _sha256_file(Path(models_dir) / f"{name}.py")
            + _sha256_file(Path(responses_path))
            + _sampler_signature(settings)
        ).encode("utf-8")
    ).hexdigest()[:16]


def cached_fit_path(cache_dir: Path, name: str, fingerprint: str) -> Path:
    """Where ``fit_model`` persists (and reads back) the fit with this fingerprint."""
    return Path(cache_dir) / f"{name}.{fingerprint}.nc"


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

    Every sampler argument defaults to ``None``, meaning "unset -- resolve it".
    :func:`resolve_fit_settings` then applies an explicit caller value first, the
    model file's own ``SAMPLER_SETTINGS`` declaration next, and the centralized
    production defaults last. Defaulting these to the production values instead
    would make "the caller wants 0.99" indistinguishable from "the caller said
    nothing", silently overriding every model-declared setting.

    If `cache_dir` is given and `<cache_dir>/<name>.<fingerprint>.nc` exists,
    load idata from disk instead of refitting.

    A fit that fails the convergence gate (:func:`convergence_problems`) is
    refit once at ``ESCALATED_TARGET_ACCEPT`` (user decision 2026-09-26), and
    that fit is returned — to every caller, since both fits are cached. A
    single-chain fit is never refit: its R-hat is undefined.
    """
    models_dir = Path(models_dir)
    responses_path = Path(responses_path)
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

    fitted = _fit_once(name, models_dir, responses_path, settings, cache_dir)
    if (
        settings["chains"] >= 2
        and settings["target_accept"] < ESCALATED_TARGET_ACCEPT
        and convergence_problems_of(fitted)
    ):
        print(
            f"  [fit] {name} did not converge at target_accept "
            f"{settings['target_accept']}; refitting at {ESCALATED_TARGET_ACCEPT}.",
            flush=True,
        )
        escalated = {**settings, "target_accept": ESCALATED_TARGET_ACCEPT}
        fitted = _fit_once(name, models_dir, responses_path, escalated, cache_dir)
    return fitted


def _fit_once(
    name: str,
    models_dir: Path,
    responses_path: Path,
    settings: Dict[str, Any],
    cache_dir: Optional[Path],
) -> FittedModel:
    """One fit at resolved ``settings``, or its cached result."""
    pm = _import_pymc()
    az = _import_arviz()
    model = load_pymc_model(name, models_dir)
    fp = fit_fingerprint(name, models_dir, responses_path, settings)

    nc_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        nc_path = cached_fit_path(cache_dir, name, fp)

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


# ---------------------------------------------------------------------------
# Sampling diagnostics
# ---------------------------------------------------------------------------

def _divergence_count(idata: Any) -> Optional[int]:
    """Number of divergent transitions, or None if the trace does not record any.

    None is a real answer, not a failure: a sampler that is not NUTS (a model
    with discrete parameters falls back to Metropolis) writes no ``diverging``
    stat. It is deliberately NOT folded into 0 -- "no divergences" and "nobody
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
    values = [float(rhat[v].max()) for v in rhat.data_vars]
    # A NaN anywhere means "unverified": Python's max() over NaNs depends on
    # their order, so it could have hidden one.
    if not values or any(math.isnan(v) for v in values):
        return float("nan")
    return max(values)


def convergence_problems(idata: Any, var_names: Sequence[str]) -> List[str]:
    """Why a fit has not converged, one line per problem; empty when it has.

    Checks the free parameters ``var_names`` for divergent transitions above
    ``MAX_DIVERGENCE_FRACTION`` of all draws (a trace that records none cannot
    be checked, which is itself a problem), R-hat above ``MAX_R_HAT``
    (undefined R-hat, e.g. one chain, counts) and bulk ESS below
    ``MIN_BULK_ESS``.
    """
    az = _import_arviz()
    problems: List[str] = []
    n_div = _divergence_count(idata)
    if n_div is None:
        problems.append("the trace records no divergence statistic, so sampling could not be checked")
    else:
        n_draws = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
        if n_div > MAX_DIVERGENCE_FRACTION * n_draws:
            problems.append(f"{n_div} divergent transitions of {n_draws}")
    if var_names:
        posterior = idata.posterior[list(var_names)]
        rhat = az.rhat(posterior)
        worst_rhat = [float(rhat[v].max()) for v in rhat.data_vars]
        if any(math.isnan(v) for v in worst_rhat):
            problems.append("R-hat is undefined (a single chain?)")
        elif max(worst_rhat) > MAX_R_HAT:
            problems.append(f"max R-hat {max(worst_rhat):.3f} > {MAX_R_HAT}")
        ess = az.ess(posterior, method="bulk")
        lowest_ess = min(float(ess[v].min()) for v in ess.data_vars)
        if not lowest_ess >= MIN_BULK_ESS:
            problems.append(f"min bulk ESS {lowest_ess:.0f} < {MIN_BULK_ESS}")
    return problems


def convergence_problems_of(fitted: "FittedModel") -> List[str]:
    """``convergence_problems`` over a fitted model's free parameters."""
    free = [rv.name for rv in fitted.model.free_RVs]
    return convergence_problems(fitted.idata, free)


def _warn_sampling_diagnostics(name: str, idata: Any) -> None:
    """Loudly surface NUTS trouble (divergences, poor R-hat) for a fit.

    These are advisory, not fatal -- ArviZ still returns usable arrays -- but a fit
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
            "may be biased -- treat its ELPD-LOO with caution.",
            file=sys.stderr,
            flush=True,
        )
    max_rhat = _max_rhat(idata)
    if not math.isfinite(max_rhat):
        print(
            f"  [warn] {name}: R-hat is unavailable (got {max_rhat}); convergence "
            "was NOT verified -- a single-chain fit cannot report one.",
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


# ---------------------------------------------------------------------------
# Parallel fitting
# ---------------------------------------------------------------------------

# Thread-count variables a fit worker pins to 1. One fit already runs one
# process per chain; a BLAS pool per chain on top of that would oversubscribe
# the allocation as soon as fits run side by side.
_SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def allocated_cpus() -> int:
    """CPUs this process may run on: the Slurm allocation on a compute node.

    ``os.cpu_count()`` reports the whole node; the scheduler affinity mask is
    what ``--cpus-per-task`` actually granted.
    """
    return len(os.sched_getaffinity(0))


def _fit_cpus(settings: Dict[str, Any]) -> int:
    """CPUs one fit occupies: PyMC runs ``min(cores, chains)`` chain processes."""
    return max(1, min(int(settings["cores"]), int(settings["chains"])))


def default_fit_workers(cpus: int, per_fit_cpus: Sequence[int]) -> int:
    """Concurrent fits that keep ``workers x cores-per-fit`` within ``cpus``.

    The widest fit sets the budget, so a batch that mixes 2- and 4-chain fits
    is sized for the 4-chain ones. Never below 1: a single fit that on its own
    exceeds the allocation still runs, as it always has.
    """
    if not per_fit_cpus:
        raise ValueError("default_fit_workers: no fits to size the pool for.")
    return max(1, int(cpus) // max(int(c) for c in per_fit_cpus))


class FitWorkerFailure(RuntimeError):
    """A fit that failed inside a pool worker, re-raised in the parent.

    Its one argument is ``"<OriginalType>: <message>"``. The original exception
    is not sent across the process boundary: an exception whose constructor
    takes extra arguments -- PyMC's ``ParallelSamplingError`` takes the chain
    number -- cannot be unpickled by the parent, and an exception the pool
    cannot deliver breaks the whole pool (``BrokenProcessPool``) instead of
    reporting one model's failure. The original traceback goes to the
    worker's stderr, which is the run log.
    """


def _fit_model_in_worker(
    name: str,
    models_dir: Path,
    responses_path: Path,
    cache_dir: Path,
    fit_kwargs: Dict[str, Any],
) -> str:
    """Fit one model inside a pool worker and hand back its cache fingerprint.

    The fit itself is :func:`fit_model` with ``cache_dir`` set, so the worker's
    only product is the ``.nc`` it persists; a ``FittedModel`` holds a compiled
    PyMC graph and never crosses a process boundary. Thread counts are pinned
    to one before sampling (``threadpoolctl`` -- a dependency of PyMC -- for
    the BLAS already loaded in this process, the environment for the chain
    processes PyMC starts next).

    PyMC's chain processes are forked from this worker. A spawned child's
    default start method is spawn, under which PyMC pickles the step method
    for its chains, and the model lives in a module ``load_pymc_model``
    executed from a file, which a fresh interpreter cannot import ("The model
    could not be unpickled"). Forking is what the sequential path does from
    the main process, and this worker has no threads for a fork to break.
    """
    multiprocessing.set_start_method("fork", force=True)
    for var in _SINGLE_THREAD_ENV:
        os.environ[var] = "1"
    from threadpoolctl import threadpool_limits

    try:
        with threadpool_limits(limits=1):
            fitted = fit_model(
                name, models_dir, responses_path, cache_dir=cache_dir, **fit_kwargs
            )
    except Exception as e:  # noqa: BLE001 — every failure must reach the parent by name
        traceback.print_exc(file=sys.stderr)
        raise FitWorkerFailure(f"{type(e).__name__}: {e}") from None
    return fitted.fingerprint


def _fit_executor(workers: int) -> ProcessPoolExecutor:
    """The pool the fits run in: fresh (spawned) interpreters, not forks.

    A worker forks PyMC's chain processes itself, so it must not be a daemon
    (``multiprocessing.Pool`` workers are; ``ProcessPoolExecutor``'s are not).
    Spawning rather than forking keeps the parent's threads, compiled graphs
    and cached InferenceData out of the children. A spawned child re-imports
    the entry script as ``__mp_main__``, so every entry point that reaches a
    fit needs the usual ``if __name__ == "__main__":`` guard (all of the
    repo's do; an ad-hoc script without one re-runs itself in each worker).
    """
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )


def _sample_models_in_pool(
    names: Sequence[str],
    models_dir: Path,
    responses_path: Path,
    cache_dir: Path,
    fit_kwargs: Dict[str, Any],
    *,
    workers: int,
    stop_on_failure: bool,
) -> Dict[str, Optional[BaseException]]:
    """Sample ``names`` concurrently, each into ``cache_dir``; report per name.

    Returns ``{name: None}`` for a model whose fit is now on disk and
    ``{name: exception}`` for one whose worker raised. With ``stop_on_failure``
    the fits not yet started when the first failure lands are cancelled and
    reported as ``CancelledError``; running ones finish and stay cached.

    Two things are the harness's fault rather than a model's and raise here:
    a worker that returned without leaving its ``.nc`` behind, and one whose
    fingerprint is not the one the parent computes for the same inputs (the
    parent would load the wrong fit, or none).
    """
    names = list(names)
    models_dir = Path(models_dir)
    responses_path = Path(responses_path)
    cache_dir = Path(cache_dir)
    expected = {
        name: fit_fingerprint(
            name, models_dir, responses_path,
            resolve_fit_settings(name, models_dir, fit_kwargs),
        )
        for name in names
    }
    outcomes: Dict[str, Optional[BaseException]] = {}
    with _fit_executor(min(workers, len(names))) as pool:
        futures = {
            pool.submit(
                _fit_model_in_worker,
                name, models_dir, responses_path, cache_dir, dict(fit_kwargs),
            ): name
            for name in names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                fingerprint = future.result()
            except CancelledError as e:
                outcomes[name] = e
                continue
            except Exception as e:  # noqa: BLE001 — the worker's failure, reported by name
                outcomes[name] = e
                if stop_on_failure:
                    for other in futures:
                        if other.cancel():
                            outcomes[futures[other]] = CancelledError(
                                f"fit of {futures[other]!r} cancelled after "
                                f"{name!r} failed"
                            )
                continue
            if fingerprint != expected[name]:
                raise RuntimeError(
                    f"fit worker for {name!r} returned fingerprint {fingerprint} "
                    f"but the parent expects {expected[name]} for the same model, "
                    "data and sampler settings."
                )
            if not cached_fit_path(cache_dir, name, fingerprint).exists():
                raise RuntimeError(
                    f"fit worker for {name!r} returned but wrote no fit at "
                    f"{cached_fit_path(cache_dir, name, fingerprint)}."
                )
            outcomes[name] = None
    return outcomes


def _fit_outcomes(
    model_names: Sequence[str],
    models_dir: Path,
    responses_path: Path,
    *,
    cache_dir: Optional[Path],
    fit_workers: Optional[int],
    fit_kwargs: Dict[str, Any],
    stop_on_failure: bool,
) -> Dict[str, Union[FittedModel, BaseException]]:
    """Fit every model, concurrently where it pays, and report each outcome.

    Cache hits (in-process, then on disk) are served in this process. The
    models that actually need MCMC are sampled in a process pool when there
    are at least two of them and the worker budget allows more than one at a
    time; the children persist their fits and the parent loads them, so the
    result is the same ``FittedModel`` a sequential fit would have produced
    (and the same cache file). Without a ``cache_dir`` the pool hands its
    fits over through a temporary directory that is removed once loaded.

    Every successful fit lands in the in-process cache. A failed fit is
    returned as its exception under the model's name; with ``stop_on_failure``
    nothing further is sampled after the first one.
    """
    if fit_workers is not None and fit_workers < 1:
        raise ValueError(f"fit_workers must be >= 1, got {fit_workers}.")
    models_dir = Path(models_dir)
    responses_path = Path(responses_path)
    outcomes: Dict[str, Union[FittedModel, BaseException]] = {}
    pending: List[str] = []
    settings: Dict[str, Dict[str, Any]] = {}
    for name in model_names:
        # Keying a fit executes the model file (for its SAMPLER_SETTINGS); a
        # file that raises is that model's failure, reported like a failed fit.
        try:
            key = _cache_key(name, models_dir, responses_path, fit_kwargs)
            settings[name] = resolve_fit_settings(name, models_dir, fit_kwargs)
        except Exception as e:  # noqa: BLE001 — reported by name; the caller decides
            outcomes[name] = e
            if stop_on_failure:
                return outcomes
            continue
        cached = _FIT_CACHE.get(key)
        if cached is not None:
            _warn_sampling_diagnostics(name, cached.idata)
            outcomes[name] = cached
        else:
            pending.append(name)

    to_sample = [
        name
        for name in pending
        if cache_dir is None
        or not cached_fit_path(
            cache_dir, name,
            fit_fingerprint(name, models_dir, responses_path, settings[name]),
        ).exists()
    ]
    workers = 1
    if len(to_sample) >= 2:
        workers = (
            fit_workers
            if fit_workers is not None
            else default_fit_workers(
                allocated_cpus(), [_fit_cpus(settings[name]) for name in to_sample]
            )
        )

    def fit_in_process(name: str, load_dir: Optional[Path]) -> None:
        try:
            fitted = fit_model(
                name, models_dir, responses_path, cache_dir=load_dir, **fit_kwargs
            )
        except Exception as e:  # noqa: BLE001 — reported by name; the caller decides
            outcomes[name] = e
            return
        _FIT_CACHE[_cache_key(name, models_dir, responses_path, fit_kwargs)] = fitted
        outcomes[name] = fitted

    if workers < 2:
        for name in pending:
            fit_in_process(name, cache_dir)
            if stop_on_failure and isinstance(outcomes[name], BaseException):
                break
        return outcomes

    with tempfile.TemporaryDirectory(prefix="pymc_fits_") as transport:
        pool_dir = Path(cache_dir) if cache_dir is not None else Path(transport)
        pool_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"  [fit] sampling {len(to_sample)} models with {min(workers, len(to_sample))} "
            f"concurrent fits (allocated CPUs: {allocated_cpus()}; "
            f"widest fit: {max(_fit_cpus(settings[n]) for n in to_sample)} cores)",
            flush=True,
        )
        sampled = _sample_models_in_pool(
            to_sample, models_dir, responses_path, pool_dir, fit_kwargs,
            workers=workers, stop_on_failure=stop_on_failure,
        )
        for name, failure in sampled.items():
            if failure is not None:
                outcomes[name] = failure
        for name in pending:
            if name in outcomes:
                continue
            fit_in_process(name, pool_dir)
    return outcomes


def fit_models_cached(
    model_names: List[str],
    models_dir: Path,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    fit_workers: Optional[int] = None,
    **fit_kwargs: Any,
) -> Dict[str, FittedModel]:
    """Fit each model in `model_names`, reusing cached fits keyed by
    (model_name, sha256(model.py), sha256(responses.csv), sampler settings). Each
    call to `pm.sample` is expensive, so identical (model, data, sampler) triples
    are reused within a process. If `cache_dir` is given, also persists/reads .nc
    files (keyed by the same triple).

    Models that need MCMC are fit concurrently (see :func:`_fit_outcomes`).
    ``fit_workers`` caps the concurrent fits; by default
    ``workers x cores-per-fit`` fills but never exceeds the CPUs this process
    is allocated (:func:`default_fit_workers`). It is not a sampler setting,
    so it takes no part in any cache key. The failed fit is raised; fits the
    pool cancelled because of it are not reported.
    """
    outcomes = _fit_outcomes(
        model_names, models_dir, responses_path,
        cache_dir=cache_dir, fit_workers=fit_workers, fit_kwargs=fit_kwargs,
        stop_on_failure=True,
    )
    failures = [o for o in outcomes.values() if isinstance(o, BaseException)]
    if failures:
        # The failure that stopped the batch, not one of the cancellations it
        # caused (a cancelled fit's name may sort before the failed one's).
        causes = [f for f in failures if not isinstance(f, CancelledError)]
        raise (causes or failures)[0]
    missing = [name for name in model_names if name not in outcomes]
    if missing:
        raise RuntimeError(f"No fit and no failure was recorded for {missing}.")
    return {name: outcomes[name] for name in model_names}  # type: ignore[misc]


def fit_models_to_cache(
    model_names: List[str],
    models_dir: Path,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    fit_workers: Optional[int] = None,
    **fit_kwargs: Any,
) -> Dict[str, str]:
    """Fit every model that can be fit; report the ones that cannot, by name.

    The tolerant sibling of :func:`fit_models_cached` for callers that screen
    a model set — the inner loop's start-of-experiment ELPD screen — and must
    keep going when one model fails: every fit that succeeds is in the cache
    afterwards, and the return value maps each model that failed to
    ``"<ExceptionType>: <message>"``. An empty dict means every model fit.
    """
    outcomes = _fit_outcomes(
        model_names, models_dir, responses_path,
        cache_dir=cache_dir, fit_workers=fit_workers, fit_kwargs=fit_kwargs,
        stop_on_failure=False,
    )
    return {
        name: _describe_failure(outcome)
        for name, outcome in outcomes.items()
        if isinstance(outcome, BaseException)
    }


def _describe_failure(failure: BaseException) -> str:
    """``"<Type>: <message>"``; a worker failure already carries its original type."""
    if isinstance(failure, FitWorkerFailure):
        return str(failure)
    return f"{type(failure).__name__}: {failure}"


def clear_fit_cache() -> None:
    """Clear the in-process fit cache. Useful for tests."""
    _FIT_CACHE.clear()


def evict_fit_cache(model_name: str) -> int:
    """Drop every cached fit for ``model_name``; return how many were evicted.

    Used when the inner loop prunes a losing model -- its InferenceData would
    otherwise stay resident in the in-process cache for the rest of the run.
    The cache key leads with the model name (see ``_cache_key``).
    """
    keys = [k for k in _FIT_CACHE if k and k[0] == model_name]
    for k in keys:
        del _FIT_CACHE[k]
    return len(keys)
