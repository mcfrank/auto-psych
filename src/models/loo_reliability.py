"""PSIS-LOO reliability: is a fit's ELPD-LOO trustworthy enough to act on?

The inner loop excludes a model from export, from pruning decisions and from
the next experiment's design prior when its PSIS-LOO estimate is "unreliable".
arviz's own verdict (``loo(...).warning``) is blanket: it fires when *any*
single observation's Pareto tail index k exceeds ``good_k`` (≈ 0.7). For the
Bernoulli choice models fit here that verdict is wrong in two ways:

1. **Exact trials.** A trial whose log-likelihood is identical across every
   posterior draw — what a clipped, saturated ``p_left`` (``pt.clip(p, 1e-6,
   1 - 1e-6)``) produces on an extreme stimulus — has importance weights that
   are all equal. There is no tail for PSIS to fit, so arviz reports k = inf,
   but the LOO term for that trial is exact: importance sampling with constant
   weights is the plain estimate. Measured on the 2026-08/09 holdout sweeps,
   every non-finite k in the excluded winners' fits was such a trial (40–720
   per model, one per participant × stimulus), and not one trial had a finite
   k above 0.7. Those winners were dropped for nothing.
2. **One trial in thousands.** PSIS's estimate at a genuinely high-k trial is
   optimistic by at most that trial's (lpd_i − elpd_i), a few nats for a
   Bernoulli likelihood. A handful of such trials cannot move a total ELPD
   over thousands of trials by more than a few nats, far inside the
   differences the loop treats as decisive. Only a real *proportion* of
   high-k trials says the model is misspecified enough that its ELPD-LOO
   cannot be trusted.

So this module (a) exempts exact trials and (b) judges the rest by the
proportion whose Pareto k is not ≤ ``good_k`` (non-finite k counts as bad
unless the trial is exact). The verdict, the counts behind it and the
pointwise ``ELPDData`` (reused by ``az.compare``, so LOO is computed once per
fit) are returned together so every consumer reads one diagnostic.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

# A fit is unreliable when more than this proportion of its trials have a
# Pareto k above good_k (exact trials excluded). 1 % is deliberately small: at
# the sweeps' 2 560–7 680 pooled trials it allows a few dozen influential
# trials — a worst-case ELPD bias of a few dozen nats, comparable to the
# smallest ELPD gaps the loop ever acts on — while a model with more high-k
# trials than that is misspecified in a way PSIS cannot paper over.
DEFAULT_BAD_K_TOLERANCE = 0.01

# A trial is "exact" when its log-likelihood varies by no more than this across
# all posterior draws (nats). A clipped p_left gives exactly 0; the slack only
# absorbs floating-point noise, never a real posterior spread.
EXACT_TRIAL_LOGLIK_SPREAD = 1e-8


@dataclass(frozen=True)
class LooDiagnostics:
    """PSIS-LOO of one fit plus the reliability verdict and its evidence."""

    elpd_loo: float
    loo: Any  # arviz ELPDData with pointwise pareto_k; reusable by az.compare
    n_points: int
    n_exact: int  # trials whose LOO term is exact (constant log-likelihood)
    n_bad_k: int  # non-exact trials whose Pareto k is not <= good_k
    frac_bad_k: float  # n_bad_k / n_points
    max_pareto_k: float  # largest FINITE k over non-exact trials (nan if none)
    good_k: float  # arviz's k threshold for this sample size
    bad_k_tolerance: float
    unreliable: bool


def _pointwise_log_likelihood(idata: Any) -> np.ndarray:
    """The fit's log-likelihood draws as ``(n_draws, n_trials)``.

    Fails loudly unless the ``log_likelihood`` group holds exactly one
    variable: with several, which one arviz's LOO scored is ambiguous and the
    exact-trial mask below could be built from the wrong one.
    """
    if not hasattr(idata, "log_likelihood"):
        raise ValueError(
            "InferenceData has no log_likelihood group; PSIS-LOO needs pointwise "
            "log-likelihood draws (fit with idata_kwargs={'log_likelihood': True})."
        )
    names = list(idata.log_likelihood.data_vars)
    if len(names) != 1:
        raise ValueError(
            "PSIS-LOO reliability needs exactly one log_likelihood variable, "
            f"got {names}."
        )
    arr = np.asarray(idata.log_likelihood[names[0]].values, dtype="float64")
    n_chains, n_draws = arr.shape[0], arr.shape[1]
    return arr.reshape(n_chains * n_draws, -1)


def loo_diagnostics(
    idata: Any, *, bad_k_tolerance: float = DEFAULT_BAD_K_TOLERANCE
) -> LooDiagnostics:
    """PSIS-LOO of ``idata`` with a reliability verdict that ignores exact trials.

    ``unreliable`` is True when the proportion of trials (over all trials)
    whose Pareto k is *not* ≤ ``good_k`` — excluding exact trials, whose LOO
    term needs no importance weighting — exceeds ``bad_k_tolerance``.
    """
    import arviz as az

    if not (0.0 <= bad_k_tolerance <= 1.0):
        raise ValueError(
            f"bad_k_tolerance must be a proportion in [0, 1], got {bad_k_tolerance!r}."
        )

    # Validate the log-likelihood group first so an ambiguous fit fails with
    # our message rather than arviz's.
    log_lik = _pointwise_log_likelihood(idata)

    with warnings.catch_warnings():
        # arviz warns whenever any k > good_k. That is exactly the blanket
        # verdict this function replaces (its exact-trial k = inf is the usual
        # trigger), so the warning would be noise here; our own verdict is the
        # one the caller reports.
        warnings.filterwarnings(
            "ignore",
            message="Estimated shape parameter of Pareto distribution",
            category=UserWarning,
        )
        loo = az.loo(idata, pointwise=True)

    pareto_k = np.asarray(loo.pareto_k, dtype="float64").reshape(-1)
    if pareto_k.shape[0] != log_lik.shape[1]:
        raise ValueError(
            f"Pareto-k count {pareto_k.shape[0]} does not match the "
            f"{log_lik.shape[1]} log-likelihood trials."
        )

    spread = log_lik.max(axis=0) - log_lik.min(axis=0)
    exact = spread <= EXACT_TRIAL_LOGLIK_SPREAD
    good_k = float(loo.good_k)
    # ``~(k <= good_k)`` so that inf and nan count as bad unless the trial is exact.
    bad = ~exact & ~(pareto_k <= good_k)
    finite_nonexact = np.isfinite(pareto_k) & ~exact
    max_pareto_k = (
        float(pareto_k[finite_nonexact].max()) if finite_nonexact.any() else float("nan")
    )
    n_points = int(pareto_k.shape[0])
    n_bad = int(bad.sum())
    frac_bad = n_bad / n_points if n_points else 0.0

    return LooDiagnostics(
        elpd_loo=float(loo.elpd_loo),
        loo=loo,
        n_points=n_points,
        n_exact=int(exact.sum()),
        n_bad_k=n_bad,
        frac_bad_k=float(frac_bad),
        max_pareto_k=max_pareto_k,
        good_k=good_k,
        bad_k_tolerance=float(bad_k_tolerance),
        unreliable=bool(frac_bad > bad_k_tolerance),
    )


def describe_unreliable(name: str, diag: LooDiagnostics) -> str:
    """One-line, number-bearing statement of why ``name``'s PSIS-LOO is unreliable."""
    return (
        f"{name}: PSIS-LOO is unreliable — {diag.n_bad_k} of {diag.n_points} trials "
        f"({diag.frac_bad_k:.1%}) have Pareto k > {diag.good_k:.2f}, above the "
        f"{diag.bad_k_tolerance:.1%} tolerance ({diag.n_exact} exact trials exempt); "
        "its ELPD-LOO may be inaccurate."
    )
