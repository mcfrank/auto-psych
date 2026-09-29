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

import atexit
import hashlib
import math
import multiprocessing
import os
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import connection as mp_connection
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

from src.models.mcmc_defaults import (
    ESCALATED_TARGET_ACCEPT,
    MAX_DIVERGENCE_FRACTION,
    MAX_R_HAT,
    MIN_BULK_ESS,
    NEAR_MISS_MAX_DIVERGENCE_FRACTION,
    NEAR_MISS_MAX_R_HAT,
    NEAR_MISS_MIN_BULK_ESS,
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


def raised_in_file(exc: BaseException, source_file: Path) -> bool:
    """Whether ``exc`` was raised in, or below, code from ``source_file``: one
    frame of its traceback runs that file's code."""
    target = Path(source_file).resolve()
    tb = exc.__traceback__
    while tb is not None:
        if Path(tb.tb_frame.f_code.co_filename).resolve() == target:
            return True
        tb = tb.tb_next
    return False


def is_model_failure(exc: BaseException, source_file: Optional[Path]) -> bool:
    """Whether ``exc`` is the failure of the model loaded from ``source_file``
    rather than of the harness or the machine.

    An infrastructure error (``INFRASTRUCTURE_ERRORS``) never is. A code error
    (``BROKEN_MODEL_CODE_ERRORS``: a ``NameError`` in a ``compute_features``
    hook, say) is the model's when it was raised in the model's own file, and
    the harness's otherwise — a broken harness must not be blamed on every
    model it touches. Any other error (a feature that is not finite, a hook
    indexing past the end of a short sequence) is the model's.
    """
    if isinstance(exc, INFRASTRUCTURE_ERRORS):
        return False
    if isinstance(exc, BROKEN_MODEL_CODE_ERRORS):
        return source_file is not None and raised_in_file(exc, source_file)
    return True


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
    data*. A broken harness is not that, and propagates — otherwise a missing
    dependency would silently condemn every candidate the inner loop
    generated. A code error (``BROKEN_MODEL_CODE_ERRORS``) raised in the
    model's own file — a typo in its ``compute_features`` — is the model's,
    and a ``(False, reason)`` like any other (``is_model_failure``); it used
    to propagate and end the cell.
    """
    pm = _import_pymc()
    model_file = Path(models_dir) / f"{name}.py"
    model = load_pymc_model(name, models_dir)
    try:
        observed = extract_observed(responses_path, model)
        with model:
            pm.set_data(observed)
    except Exception as e:
        if not is_model_failure(e, model_file):
            raise
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
        if not is_model_failure(e, model_file):
            raise
        return False, f"logp evaluation raised: {type(e).__name__}: {e}"
    if not math.isfinite(logp):
        return False, f"non-finite logp ({logp}) at the initial point"
    try:
        grad = np.asarray(model.compile_dlogp()(point), dtype=float)
    except Exception as e:
        if not is_model_failure(e, model_file):
            raise
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
        arr = _validated_p_left_draws(
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


class InvalidPredictions(ValueError):
    """Predictive p_left that is not a probability (NaN, or outside [0, 1]).
    ``draws`` holds the (draw, stimulus) predictions with every invalid value
    set to NaN, so a caller can tell exactly which stimuli are affected: the
    recovery evaluation excludes and logs them, the design screens the model
    out (``design/screened_out.json``), the novelty gate rejects a candidate
    with the reason. Every other caller fails loudly."""

    def __init__(self, message: str, draws: np.ndarray) -> None:
        super().__init__(message)
        self.draws = draws

    def invalid_stimuli(self) -> np.ndarray:
        """Boolean mask over stimuli: True where any draw is not a probability."""
        return np.isnan(self.draws).any(axis=0)


def _validated_p_left_draws(raw: Any, *, context: str) -> np.ndarray:
    """``raw`` (..., n_stim) as floats; ``InvalidPredictions`` if any value is
    not a probability."""
    try:
        return validate_probability_array(raw, context=context)
    except ValueError as exc:
        raw = np.asarray(raw, dtype=float)
        bad = ~np.isfinite(raw) | (raw < 0) | (raw > 1)
        masked = np.where(bad, np.nan, raw).reshape(-1, raw.shape[-1])
        raise InvalidPredictions(str(exc), masked) from exc


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
        arr = _validated_p_left_draws(
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
    time_limit_sec: Optional[float] = None,
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

    A fit that fails the convergence gate (:func:`convergence_problems`) as a
    near miss (:func:`is_near_miss`) is refit once at
    ``ESCALATED_TARGET_ACCEPT`` (user decisions 2026-09-26, 2026-09-27), and
    that fit is returned — to every caller, since both fits are cached. A fit
    far from converging is returned as it is, and fails the gate with its own
    numbers. A single-chain fit is never refit: its R-hat is undefined.

    ``time_limit_sec`` (candidate admission: ``CANDIDATE_FIT_TIME_LIMIT_SEC``)
    runs each sampling run — the first fit and a refit — in its own process
    session and stops it, chains and all, when it is still sampling at the
    limit, raising :class:`FitTimeLimitExceeded`. A model's own failure then
    arrives as :class:`FitWorkerFailure`. Cached fits load as usual, and a
    time-limited fit that failed is remembered for the rest of the process
    (:func:`sample_fits_time_limited`), so asking again does not re-sample it.
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

    fitted = _fit_once_within(name, models_dir, responses_path, settings, cache_dir, time_limit_sec)
    if _refit_decision(name, fitted, settings):
        fitted = _fit_once_within(
            name, models_dir, responses_path, refit_settings(settings, fitted.fingerprint),
            cache_dir, time_limit_sec,
        )
    return fitted


def refit_settings(settings: Dict[str, Any], first_fingerprint: str) -> Dict[str, Any]:
    """The settings of a near miss's refit: ``ESCALATED_TARGET_ACCEPT`` and a
    random seed of its own.

    The refit used to reuse the first fit's ``random_seed`` (42, for every fit
    in every cell), so it started its chains from the same draws. Its seed is
    now derived from the first fit's seed and fingerprint (model source, data
    and settings; ``refit_random_seed``): different from the first fit's,
    different for every model and data set, and the same on every resume.
    ``random_seed`` is a sampler setting, so it is part of the refit's cache
    fingerprint.
    """
    return {
        **settings,
        "target_accept": ESCALATED_TARGET_ACCEPT,
        "random_seed": refit_random_seed(settings["random_seed"], first_fingerprint),
    }


def refit_random_seed(base_seed: int, first_fingerprint: str) -> int:
    """A refit's random seed, from everything that identifies its first fit
    (the style of the harness's ``derive_seed``); never the first fit's own."""
    digest = hashlib.sha256(f"{base_seed}|{first_fingerprint}|refit".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big") % 2**31
    return seed if seed != base_seed else (seed + 1) % 2**31


def _fit_once_within(
    name: str,
    models_dir: Path,
    responses_path: Path,
    settings: Dict[str, Any],
    cache_dir: Optional[Path],
    time_limit_sec: Optional[float],
) -> FittedModel:
    """``_fit_once``, with any sampling it needs stopped at ``time_limit_sec``.

    The sampling runs in a child process that persists the fit, which is then
    loaded here like any cached fit. Without a ``cache_dir`` the child writes
    to a temporary directory and the fit is loaded into memory before the
    directory goes.
    """
    if time_limit_sec is None:
        return _fit_once(name, models_dir, responses_path, settings, cache_dir)
    if cache_dir is not None:
        # A remembered failure or timeout comes back before a cached file is
        # looked at: a child killed at its limit just after writing its fit
        # still timed out.
        request = TimeLimitedFit(name, models_dir, responses_path, settings, Path(cache_dir))
        _raise_failure(sample_fits_time_limited([request], time_limit_sec=time_limit_sec)[0])
        return _fit_once(name, models_dir, responses_path, settings, cache_dir)
    with tempfile.TemporaryDirectory(prefix="pymc_fit_") as transport:
        request = TimeLimitedFit(name, models_dir, responses_path, settings, Path(transport))
        _raise_failure(sample_fits_time_limited([request], time_limit_sec=time_limit_sec)[0])
        with _import_arviz().rc_context(rc={"data.load": "eager"}):
            return _fit_once(name, models_dir, responses_path, settings, Path(transport))


def _raise_failure(outcome: Optional[BaseException]) -> None:
    if outcome is not None:
        raise outcome


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
        try:
            idata = az.from_netcdf(str(nc_path))
        except Exception as e:  # noqa: BLE001 — any unreadable cache file is the harness's fault
            raise FitInfrastructureFailure(
                f"the cached fit {nc_path} cannot be read ({type(e).__name__}: {e}). "
                "It is corrupt (a write cut short before fits were written "
                f"atomically?), not a property of the model {name!r}; delete it "
                "to refit."
            ) from e
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
        write_fit_file(idata, nc_path)

    return FittedModel(name=name, model=model, idata=idata, fingerprint=fp)


def write_fit_file(idata: Any, nc_path: Path) -> None:
    """Persist ``idata`` at ``nc_path`` atomically.

    The fit is written to a temporary file beside ``nc_path`` and renamed into
    place, so a process killed mid-write (an out-of-memory kill, a time limit)
    leaves no ``.nc`` at all rather than a truncated one that every later run
    would find in the cache. The temporary name does not end in ``.nc``.
    """
    nc_path = Path(nc_path)
    partial = nc_path.with_name(f".{nc_path.name}.{os.getpid()}.partial")
    try:
        idata.to_netcdf(str(partial))
        os.replace(partial, nc_path)
    finally:
        partial.unlink(missing_ok=True)


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


@dataclass(frozen=True)
class ConvergenceDiagnostics:
    """The numbers the convergence gate and the near-miss rule judge a fit by.

    ``n_divergent`` is None when the trace records no divergence statistic;
    ``max_r_hat`` is NaN when R-hat is undefined (a single chain) and
    ``min_bulk_ess`` NaN when no free parameter was checked.
    """

    n_divergent: Optional[int]
    n_draws: int
    max_r_hat: float
    min_bulk_ess: float

    @property
    def divergent_fraction(self) -> float:
        if self.n_divergent is None:
            return float("nan")
        return self.n_divergent / self.n_draws


def convergence_diagnostics(idata: Any, var_names: Sequence[str]) -> ConvergenceDiagnostics:
    """Divergences, worst R-hat and lowest bulk ESS over the free parameters ``var_names``."""
    az = _import_arviz()
    n_draws = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    max_r_hat = min_bulk_ess = float("nan")
    if var_names:
        posterior = idata.posterior[list(var_names)]
        rhat = az.rhat(posterior)
        worst_rhat = [float(rhat[v].max()) for v in rhat.data_vars]
        max_r_hat = float("nan") if any(math.isnan(v) for v in worst_rhat) else max(worst_rhat)
        ess = az.ess(posterior, method="bulk")
        min_bulk_ess = min(float(ess[v].min()) for v in ess.data_vars)
    return ConvergenceDiagnostics(
        n_divergent=_divergence_count(idata),
        n_draws=n_draws,
        max_r_hat=max_r_hat,
        min_bulk_ess=min_bulk_ess,
    )


def convergence_problems(idata: Any, var_names: Sequence[str]) -> List[str]:
    """Why a fit has not converged, one line per problem; empty when it has.

    Checks the free parameters ``var_names`` for divergent transitions above
    ``MAX_DIVERGENCE_FRACTION`` of all draws (a trace that records none cannot
    be checked, which is itself a problem), R-hat above ``MAX_R_HAT``
    (undefined R-hat, e.g. one chain, counts) and bulk ESS below
    ``MIN_BULK_ESS``.
    """
    diag = convergence_diagnostics(idata, var_names)
    problems: List[str] = []
    if diag.n_divergent is None:
        problems.append("the trace records no divergence statistic, so sampling could not be checked")
    elif diag.n_divergent > MAX_DIVERGENCE_FRACTION * diag.n_draws:
        problems.append(f"{diag.n_divergent} divergent transitions of {diag.n_draws}")
    if var_names:
        if math.isnan(diag.max_r_hat):
            problems.append("R-hat is undefined (a single chain?)")
        elif diag.max_r_hat > MAX_R_HAT:
            problems.append(f"max R-hat {diag.max_r_hat:.3f} > {MAX_R_HAT}")
        if not diag.min_bulk_ess >= MIN_BULK_ESS:
            problems.append(f"min bulk ESS {diag.min_bulk_ess:.0f} < {MIN_BULK_ESS}")
    return problems


def convergence_problems_of(fitted: "FittedModel") -> List[str]:
    """``convergence_problems`` over a fitted model's free parameters."""
    free = [rv.name for rv in fitted.model.free_RVs]
    return convergence_problems(fitted.idata, free)


def is_near_miss(diag: ConvergenceDiagnostics) -> bool:
    """Whether a fit that failed the convergence gate is close enough for a
    refit with smaller NUTS steps to plausibly pass it (thresholds and their
    rationale in ``src/models/mcmc_defaults.py``). Without a divergence
    statistic or an R-hat there is nothing to judge: not a near miss."""
    return (
        diag.n_divergent is not None
        and diag.divergent_fraction <= NEAR_MISS_MAX_DIVERGENCE_FRACTION
        and diag.max_r_hat <= NEAR_MISS_MAX_R_HAT
        and diag.min_bulk_ess >= NEAR_MISS_MIN_BULK_ESS
    )


def convergence_diagnostics_of(fitted: "FittedModel") -> ConvergenceDiagnostics:
    """``convergence_diagnostics`` over a fitted model's free parameters."""
    return convergence_diagnostics(fitted.idata, [rv.name for rv in fitted.model.free_RVs])


def _refit_decision(name: str, fitted: "FittedModel", settings: Dict[str, Any]) -> bool:
    """True when ``fit_model`` refits ``fitted`` at ``ESCALATED_TARGET_ACCEPT``:
    a multi-chain fit below that target_accept that failed the convergence
    gate as a near miss. Says why out loud either way when the fit failed."""
    if settings["chains"] < 2 or settings["target_accept"] >= ESCALATED_TARGET_ACCEPT:
        return False
    problems = convergence_problems_of(fitted)
    if not problems:
        return False
    if is_near_miss(convergence_diagnostics_of(fitted)):
        print(
            f"  [fit] {name} did not converge at target_accept "
            f"{settings['target_accept']} ({'; '.join(problems)}), a near miss; "
            f"refitting at {ESCALATED_TARGET_ACCEPT}.",
            flush=True,
        )
        return True
    print(
        f"  [fit] {name} did not converge at target_accept "
        f"{settings['target_accept']} ({'; '.join(problems)}), too far from "
        "converging for smaller steps to help; not refitting.",
        flush=True,
    )
    return False


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


class FitInfrastructureFailure(RuntimeError):
    """A fit that failed for a reason that is not the model's.

    A pool worker that died (one out-of-memory kill breaks the whole pool, so
    every pending fit fails with it), a fit file that cannot be read or
    written, the machine out of memory. Always raised, never reported as the
    model's failure: the start-of-experiment screen used to turn each one into
    a dropped model — starting models included — and a ledger line "MCMC fit
    failed", which later agents then read as a property of the model.
    """


# Failures of the machinery rather than of a model; see FitInfrastructureFailure.
INFRASTRUCTURE_ERRORS = (OSError, MemoryError, BrokenProcessPool, FitInfrastructureFailure)


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


class FitTimeLimitExceeded(RuntimeError):
    """A time-limited fit still sampling at its limit, and stopped there."""

    def __init__(self, name: str, limit_sec: float, target_accept: float) -> None:
        super().__init__(
            f"the fit of {name!r} (target_accept {target_accept:g}) was still sampling "
            f"after the {limit_sec / 60:g}-minute limit and was stopped"
        )
        self.name = name
        self.limit_sec = limit_sec
        self.target_accept = target_accept


@dataclass(frozen=True)
class TimeLimitedFit:
    """One sampling run for :func:`sample_fits_time_limited`: the model
    ``models_dir/<name>.py`` fit on ``responses_path`` at the resolved
    ``settings``, persisted in ``cache_dir``."""

    name: str
    models_dir: Path
    responses_path: Path
    settings: Dict[str, Any]
    cache_dir: Path

    def fingerprint(self) -> str:
        return fit_fingerprint(self.name, self.models_dir, self.responses_path, self.settings)

    def nc_path(self) -> Path:
        return cached_fit_path(self.cache_dir, self.name, self.fingerprint())


# Failures of time-limited sampling runs, by (model name, fit fingerprint,
# time limit): a run that failed or ran out of time is not sampled again in
# this process.
_FAILED_TIME_LIMITED_FITS: Dict[tuple, BaseException] = {}


@contextmanager
def _fit_process_caches() -> Iterator[str]:
    """A temporary directory under which every fit process gets a cache
    directory of its own (``_use_own_cache_dir``), removed afterwards.

    On import, arviz 0.23 writes a once-a-day marker, ``<XDG cache>/arviz/
    daily_warning``, through a fixed-name ``daily_warning.tmp`` whenever the
    stored date is not today. Fit processes that import it together after
    midnight collide on that file (one finds it already moved and raises
    ``FileNotFoundError``), which ended 11 of 24 cells of a sweep.
    """
    with tempfile.TemporaryDirectory(prefix="fit-process-caches-") as root:
        yield root


def _use_own_cache_dir(cache_root: str) -> None:
    """Point ``XDG_CACHE_HOME`` at a new directory of its own under
    ``cache_root``, before anything imports arviz."""
    if "arviz" in sys.modules:
        raise RuntimeError(
            "arviz was imported in this fit process before it got a cache directory "
            "of its own; arviz's once-a-day marker would be written to the shared one."
        )
    own = Path(cache_root) / str(os.getpid())
    own.mkdir()
    os.environ["XDG_CACHE_HOME"] = str(own)


# PyTensor compiles a model's C code in its compile directory under one file
# lock (``<compiledir>/.lock``), and a process gives up after waiting
# ``compile__timeout`` (120 s) for it. Every process of a node used to share one
# directory (``$L_SCRATCH`` is per user and node, not per job), so the
# concurrent fit processes of a cell, and those of the other cells on the node,
# queued on one lock, and a cell died on ``Timeout: The file lock
# '.../compiledir_.../.lock' could not be acquired`` (2026-09-28). Each fit
# process therefore compiles in a directory no other running process uses: a
# numbered *slot* under this process's own root, held while the fit process
# runs and handed to the next one after it, so compiled code is reused within
# a cell (a cold compile costs about 50 s a fit, a warm one a few).
_COMPILE_SLOTS_LOCK = threading.Lock()
_COMPILE_SLOTS_IN_USE: set = set()
_COMPILE_SLOTS_ROOT: Optional[Path] = None


def _compile_slots_root() -> Path:
    """This process's root of fit-process compile directories: a new directory
    under its own PyTensor ``base_compiledir``, removed when it exits."""
    global _COMPILE_SLOTS_ROOT
    if _COMPILE_SLOTS_ROOT is None:
        from pytensor import config as pytensor_config

        base = Path(pytensor_config.base_compiledir)
        base.mkdir(parents=True, exist_ok=True)
        _COMPILE_SLOTS_ROOT = Path(tempfile.mkdtemp(
            prefix=f"fit-compiledirs-{socket.gethostname()}-{os.getpid()}-", dir=base
        ))
        atexit.register(shutil.rmtree, _COMPILE_SLOTS_ROOT, True)
    return _COMPILE_SLOTS_ROOT


@contextmanager
def _compile_dirs(n: int) -> Iterator[List[str]]:
    """``n`` compile directories that no other fit process of this process
    uses until the block ends (see ``_COMPILE_SLOTS_LOCK``)."""
    root = _compile_slots_root()
    with _COMPILE_SLOTS_LOCK:
        slots = []
        slot = 0
        while len(slots) < n:
            if slot not in _COMPILE_SLOTS_IN_USE:
                slots.append(slot)
            slot += 1
        _COMPILE_SLOTS_IN_USE.update(slots)
    try:
        directories = [root / f"slot-{slot}" for slot in slots]
        for directory in directories:
            directory.mkdir(exist_ok=True)
        yield [str(directory) for directory in directories]
    finally:
        with _COMPILE_SLOTS_LOCK:
            _COMPILE_SLOTS_IN_USE.difference_update(slots)


def _use_own_compile_dir(compile_dir: str) -> None:
    """Make ``compile_dir`` this process's PyTensor ``base_compiledir``, before
    anything imports pytensor; the process's other PyTensor flags stay."""
    if "pytensor" in sys.modules:
        raise RuntimeError(
            "pytensor was imported in this fit process before it got a compile "
            "directory of its own; it would compile in the shared one."
        )
    os.environ["PYTENSOR_FLAGS"] = _flags_with_base_compiledir(
        os.environ.get("PYTENSOR_FLAGS", ""), compile_dir
    )
    from pytensor import config as pytensor_config

    if not Path(pytensor_config.compiledir).is_relative_to(compile_dir):
        raise RuntimeError(
            f"this fit process compiles in {pytensor_config.compiledir}, not under its "
            f"own {compile_dir} (a .pytensorrc setting compiledir?)."
        )


def _flags_with_base_compiledir(flags: str, compile_dir: str) -> str:
    """``PYTENSOR_FLAGS`` value ``flags`` with its ``base_compiledir`` set to
    ``compile_dir`` and every other flag kept."""
    entries = [f for f in flags.split(",") if f.strip()]
    names = [f.split("=", 1)[0].strip() for f in entries]
    if "compiledir" in names:
        raise RuntimeError(
            "PYTENSOR_FLAGS sets compiledir, which overrides the base_compiledir "
            "each fit process gets; set base_compiledir instead."
        )
    kept = [f for f, name in zip(entries, names) if name != "base_compiledir"]
    return ",".join([*kept, f"base_compiledir={compile_dir}"])


def _use_own_dirs(cache_root: str, compile_dir: str) -> None:
    """First step of every fit process: a cache directory and a PyTensor
    compile directory of its own."""
    _use_own_cache_dir(cache_root)
    _use_own_compile_dir(compile_dir)


def _use_own_dirs_from_queue(cache_root: str, compile_dirs: Any) -> None:
    """Pool-worker initializer: ``_use_own_dirs`` with the next compile
    directory from ``compile_dirs``, which holds one per worker."""
    if compile_dirs.empty():
        raise RuntimeError("a fit-pool worker started with no compile directory left for it.")
    _use_own_dirs(cache_root, compile_dirs.get())


def _run_with_own_dirs(cache_root: str, compile_dir: str, target: Any, *args: Any) -> None:
    """Entry point of a time-limited fit process: ``target(*args)`` with a
    cache directory and a compile directory of its own."""
    _use_own_dirs(cache_root, compile_dir)
    target(*args)


def _sample_in_own_session(
    name: str,
    models_dir: Path,
    responses_path: Path,
    settings: Dict[str, Any],
    cache_dir: Path,
    sender: Any,
) -> None:
    """Child process of :func:`sample_fits_time_limited`: sample one fit into
    ``cache_dir`` and report ``("ok" | "model" | "infrastructure", detail)``.

    It starts a new session first, so that its process group — this process
    and the chain processes PyMC forks from it (forked, as in
    ``_fit_model_in_worker``) — can be stopped as one. Thread counts are
    pinned to one per process, as in the fit pool.
    """
    os.setsid()
    multiprocessing.set_start_method("fork", force=True)
    for var in _SINGLE_THREAD_ENV:
        os.environ[var] = "1"
    from threadpoolctl import threadpool_limits

    try:
        with threadpool_limits(limits=1):
            _fit_once(name, Path(models_dir), Path(responses_path), settings, Path(cache_dir))
    except INFRASTRUCTURE_ERRORS as e:
        traceback.print_exc(file=sys.stderr)
        sender.send(("infrastructure", f"{type(e).__name__}: {e}"))
        return
    except Exception as e:  # noqa: BLE001 — every failure must reach the parent by name
        traceback.print_exc(file=sys.stderr)
        sender.send(("model", f"{type(e).__name__}: {e}"))
        return
    sender.send(("ok", ""))


def _stop_fit_process(process: Any, request: TimeLimitedFit) -> None:
    """Kill a fit's process group (the fit and its chains) and remove the
    temporary file of a write it cut short (``write_fit_file``)."""
    try:
        if os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGKILL)
        else:  # killed before it could start its session: no chains yet
            process.kill()
    except ProcessLookupError:
        pass
    process.join()
    for partial in Path(request.cache_dir).glob(f".{request.name}.*.nc.{process.pid}.partial"):
        partial.unlink(missing_ok=True)


def _fit_process_outcome(
    process: Any, receiver: Any, request: TimeLimitedFit
) -> Optional[BaseException]:
    """What a finished fit process reported: None when its fit is on disk, a
    ``FitWorkerFailure`` for the model's own failure. Anything else raises."""
    process.join()
    try:
        message = receiver.recv() if receiver.poll() else None
    except EOFError:  # the process closed its end without sending
        message = None
    if message is None or process.exitcode != 0:
        raise FitInfrastructureFailure(
            f"the fit process of {request.name!r} exited with code {process.exitcode} "
            "without reporting an outcome (killed, e.g. out of memory?); that is not "
            "the model's failure."
        )
    kind, detail = message
    if kind == "infrastructure":
        raise FitInfrastructureFailure(f"the fit of {request.name!r} failed: {detail}")
    if kind == "model":
        return FitWorkerFailure(detail)
    if not request.nc_path().exists():
        raise FitInfrastructureFailure(
            f"the fit process of {request.name!r} reported success but wrote no fit "
            f"at {request.nc_path()}."
        )
    return None


def sample_fits_time_limited(
    requests: Sequence[TimeLimitedFit],
    *,
    time_limit_sec: float,
    workers: int = 1,
    _target: Any = None,
) -> List[Optional[BaseException]]:
    """Sample each request in its own process, ``workers`` at a time, each
    stopped at ``time_limit_sec`` of wall-clock time; report per request.

    Returns, in request order, None for a fit now on disk, a
    ``FitWorkerFailure`` for a model whose sampling raised, and a
    ``FitTimeLimitExceeded`` for one still sampling at the limit — whose
    process group (the fit and its chain processes) is killed, so the
    sampling really stops, and whose half-written file, if any, is removed.
    Failures are remembered by (name, fingerprint, limit) and returned again
    without sampling, before any file on disk is looked at. An infrastructure failure (a process that died without
    reporting, an ``OSError``/``MemoryError`` inside one) raises, after every
    other running fit is stopped.

    ``workers`` should keep ``workers x chains`` within the allocated CPUs
    (``default_fit_workers``): the limit is wall-clock time, so an
    oversubscribed machine would stop fits that would finish on their own
    cores. ``_target`` replaces the child's entry point (tests only).
    """
    if time_limit_sec <= 0:
        raise ValueError(f"time_limit_sec must be > 0, got {time_limit_sec}.")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}.")
    outcomes: List[Optional[BaseException]] = [None] * len(requests)
    queue: List[int] = []
    for i, request in enumerate(requests):
        known = _FAILED_TIME_LIMITED_FITS.get((request.name, request.fingerprint(), time_limit_sec))
        if known is not None:
            outcomes[i] = known
        elif not request.nc_path().exists():
            queue.append(i)
    context = multiprocessing.get_context("spawn")
    # sentinel -> (index, process, receiver, deadline, compile dir)
    running: Dict[Any, tuple] = {}
    with _fit_process_caches() as cache_root, _compile_dirs(workers) as free_compile_dirs:
        try:
            while queue or running:
                while queue and len(running) < workers:
                    i = queue.pop(0)
                    request = requests[i]
                    Path(request.cache_dir).mkdir(parents=True, exist_ok=True)
                    receiver, sender = context.Pipe(duplex=False)
                    compile_dir = free_compile_dirs.pop()
                    process = context.Process(
                        target=_run_with_own_dirs,
                        args=(
                            cache_root, compile_dir, _target or _sample_in_own_session,
                            request.name, request.models_dir, request.responses_path,
                            request.settings, request.cache_dir, sender,
                        ),
                        name=f"fit-{request.name}",
                    )
                    process.start()
                    sender.close()
                    running[process.sentinel] = (
                        i, process, receiver, time.monotonic() + time_limit_sec, compile_dir,
                    )
                next_deadline = min(entry[3] for entry in running.values())
                finished = mp_connection.wait(
                    list(running), timeout=max(0.0, next_deadline - time.monotonic())
                )
                for sentinel in finished:
                    i, process, receiver, _, compile_dir = running.pop(sentinel)
                    free_compile_dirs.append(compile_dir)
                    outcomes[i] = _fit_process_outcome(process, receiver, requests[i])
                now = time.monotonic()
                for sentinel, (i, process, receiver, deadline, compile_dir) in list(running.items()):
                    if now >= deadline:
                        del running[sentinel]
                        _stop_fit_process(process, requests[i])
                        free_compile_dirs.append(compile_dir)
                        request = requests[i]
                        print(
                            f"  [fit] {request.name}: still sampling after the "
                            f"{time_limit_sec / 60:g}-minute limit; stopped.",
                            flush=True,
                        )
                        outcomes[i] = FitTimeLimitExceeded(
                            request.name, time_limit_sec, float(request.settings["target_accept"])
                        )
        finally:
            for i, process, _, _, _ in running.values():
                _stop_fit_process(process, requests[i])
    for request, outcome in zip(requests, outcomes):
        if outcome is not None:
            _FAILED_TIME_LIMITED_FITS[(request.name, request.fingerprint(), time_limit_sec)] = outcome
    return outcomes


def fit_time_limited_concurrently(
    names: Sequence[str],
    models_dir: Path,
    responses_path: Path,
    *,
    cache_dir: Path,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    time_limit_sec: float,
    workers: Optional[int] = None,
) -> None:
    """Sample, concurrently, what ``fit_model(name, …, time_limit_sec=…)``
    would sample for each of ``names``, one after another.

    First every first fit (``sample_fits_time_limited``), then every refit the
    near-miss rule asks for (``_refit_decision`` on the loaded first fit),
    each run limited to ``time_limit_sec``. Fits land in ``cache_dir`` and
    failures and timeouts are remembered, so a later ``fit_model`` call with
    the same settings and limit loads or re-raises exactly what it would have
    produced by sampling itself. ``workers`` (None ⇒ ``default_fit_workers``
    over the allocated CPUs) bounds the concurrent fits so that none of them is
    slowed past its limit by oversubscription. Infrastructure failures raise.
    """
    cache_dir = Path(cache_dir)
    first = [
        TimeLimitedFit(
            name, Path(models_dir), Path(responses_path),
            resolve_fit_settings(name, models_dir, fit_kwargs), cache_dir,
        )
        for name in names
    ]
    if not first:
        return
    if workers is None:
        workers = default_fit_workers(allocated_cpus(), [_fit_cpus(r.settings) for r in first])
    outcomes = sample_fits_time_limited(first, time_limit_sec=time_limit_sec, workers=workers)
    refits = []
    for request, outcome in zip(first, outcomes):
        if outcome is not None:
            continue
        fitted = _fit_once(request.name, request.models_dir, request.responses_path, request.settings, cache_dir)
        if _refit_decision(request.name, fitted, request.settings):
            refits.append(
                TimeLimitedFit(
                    request.name, request.models_dir, request.responses_path,
                    refit_settings(request.settings, fitted.fingerprint), cache_dir,
                )
            )
    if refits:
        sample_fits_time_limited(refits, time_limit_sec=time_limit_sec, workers=workers)


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
    except INFRASTRUCTURE_ERRORS as e:
        traceback.print_exc(file=sys.stderr)
        raise FitInfrastructureFailure(f"{type(e).__name__}: {e}") from None
    except Exception as e:  # noqa: BLE001 — every failure must reach the parent by name
        traceback.print_exc(file=sys.stderr)
        raise FitWorkerFailure(f"{type(e).__name__}: {e}") from None
    return fitted.fingerprint


def _fit_executor(
    workers: int, cache_root: str, compile_dirs: Sequence[str]
) -> ProcessPoolExecutor:
    """The pool the fits run in: fresh (spawned) interpreters, not forks, each
    with a cache directory of its own under ``cache_root`` (``_fit_process_caches``)
    and one of ``compile_dirs`` (``_compile_dirs``, one per worker; a pool never
    replaces a worker — one that dies breaks it).

    A worker forks PyMC's chain processes itself, so it must not be a daemon
    (``multiprocessing.Pool`` workers are; ``ProcessPoolExecutor``'s are not).
    Spawning rather than forking keeps the parent's threads, compiled graphs
    and cached InferenceData out of the children. A spawned child re-imports
    the entry script as ``__mp_main__``, so every entry point that reaches a
    fit needs the usual ``if __name__ == "__main__":`` guard (all of the
    repo's do; an ad-hoc script without one re-runs itself in each worker).
    """
    if len(compile_dirs) < workers:
        raise ValueError(f"{workers} fit-pool workers need as many compile directories.")
    context = multiprocessing.get_context("spawn")
    handed_out = context.SimpleQueue()
    for compile_dir in compile_dirs:
        handed_out.put(compile_dir)
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_use_own_dirs_from_queue,
        initargs=(cache_root, handed_out),
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

    Only a model's own failure (``FitWorkerFailure``) is reported by name.
    Everything else is the harness's fault rather than a model's and raises
    ``FitInfrastructureFailure`` here: a broken pool (a worker killed, e.g.
    out of memory, fails every pending fit), an infrastructure error inside a
    worker, a worker that returned without leaving its ``.nc`` behind, and
    one whose fingerprint is not one the parent computes for the same inputs,
    at the loop's target_accept or the refit's (the parent would load the
    wrong fit, or none).
    """
    names = list(names)
    models_dir = Path(models_dir)
    responses_path = Path(responses_path)
    cache_dir = Path(cache_dir)
    # A worker whose first fit fails the convergence gate as a near miss
    # returns the refit (fit_model: ESCALATED_TARGET_ACCEPT and a seed of its
    # own, refit_settings); both are on disk.
    expected = {}
    for name in names:
        settings = resolve_fit_settings(name, models_dir, fit_kwargs)
        first = fit_fingerprint(name, models_dir, responses_path, settings)
        refit = fit_fingerprint(
            name, models_dir, responses_path, refit_settings(settings, first)
        )
        expected[name] = {first, refit}
    outcomes: Dict[str, Optional[BaseException]] = {}
    n_workers = min(workers, len(names))
    with _fit_process_caches() as cache_root, _compile_dirs(n_workers) as compile_dirs, \
            _fit_executor(n_workers, cache_root, compile_dirs) as pool:
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
            except FitWorkerFailure as e:  # the model's failure, reported by name
                outcomes[name] = e
                if stop_on_failure:
                    for other in futures:
                        if other.cancel():
                            outcomes[futures[other]] = CancelledError(
                                f"fit of {futures[other]!r} cancelled after "
                                f"{name!r} failed"
                            )
                continue
            except Exception as e:
                raise FitInfrastructureFailure(
                    f"the fit of {name!r} failed for a reason that is not the "
                    f"model's ({type(e).__name__}: {e}); no model of this batch "
                    "is reported as failed because of it."
                ) from e
            if fingerprint not in expected[name]:
                raise RuntimeError(
                    f"fit worker for {name!r} returned fingerprint {fingerprint} "
                    f"but the parent expects one of {sorted(expected[name])} for "
                    "the same model, data and sampler settings (or their refit)."
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
    nothing further is sampled after the first one. A failure that is not the
    model's (``INFRASTRUCTURE_ERRORS``: a broken pool, an unreadable cache
    file, out of memory) raises instead, whatever the caller.
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
        except INFRASTRUCTURE_ERRORS:
            raise
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
        except INFRASTRUCTURE_ERRORS:
            raise
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
    Only a model's own failure is reported; an infrastructure failure
    (``FitInfrastructureFailure``, ``INFRASTRUCTURE_ERRORS``) raises.
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
    """Clear the in-process fit cache (and remembered time-limited failures).
    Useful for tests."""
    _FIT_CACHE.clear()
    _FAILED_TIME_LIMITED_FITS.clear()


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
