"""Trial-level PSIS-LOO from pattern-level log-likelihood (`src.rsa.fit`).

A fit stores one log-likelihood column per pattern (display, chosen class)
and maps each trial to its pattern (``constant_data.trial_pattern``). Trials
with one pattern have identical columns, so their PSIS-LOO terms and Pareto
k are identical: PSIS runs once per pattern and the result is expanded to
trials. ELPD-LOO, its SE, p_loo, every pointwise term (the clustered SEs of
pruning and held-out comparison read them) and the reliability verdict of
`src.models.loo_reliability` (proportions over trials) are what a per-trial
computation gives.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import xarray as xr
from scipy.special import logsumexp

from src.models.loo_reliability import (
    DEFAULT_BAD_K_TOLERANCE,
    LooDiagnostics,
    _exact_trial_tolerance,
    _pointwise_log_likelihood,
)

TRIAL_DIM = "trial"


def trial_loo(idata: Any, *, bad_k_tolerance: float = DEFAULT_BAD_K_TOLERANCE) -> LooDiagnostics:
    import arviz as az

    if "constant_data" not in idata.groups() or "trial_pattern" not in idata.constant_data:
        raise ValueError(
            "this fit has no constant_data.trial_pattern: it predates pattern-level "
            "log-likelihood (src.rsa.fit); refit it"
        )
    trial_pattern = np.asarray(idata.constant_data["trial_pattern"].values, dtype=np.int64)
    ll = _pointwise_log_likelihood(idata)  # (draws, patterns)
    if trial_pattern.size and trial_pattern.max() >= ll.shape[1]:
        raise ValueError(f"trial_pattern indexes {trial_pattern.max() + 1} patterns; the fit has {ll.shape[1]}")
    with warnings.catch_warnings():
        # arviz's blanket any-k warning; the verdict below is the one reported.
        warnings.filterwarnings("ignore", message="Estimated shape parameter of Pareto distribution",
                                category=UserWarning)
        per_pattern = az.loo(idata, pointwise=True)
    loo_p = np.asarray(per_pattern["loo_i"].values, dtype=np.float64).reshape(-1)
    k_p = np.asarray(per_pattern["pareto_k"].values, dtype=np.float64).reshape(-1)
    lppd_p = logsumexp(ll, axis=0) - np.log(ll.shape[0])
    exact_p = (ll.max(axis=0) - ll.min(axis=0)) <= _exact_trial_tolerance(idata, ll)

    loo_i, k, exact = loo_p[trial_pattern], k_p[trial_pattern], exact_p[trial_pattern]
    n = len(trial_pattern)
    good_k = float(per_pattern["good_k"])
    elpd = float(loo_i.sum())
    full = per_pattern.copy()
    full["elpd_loo"] = elpd
    full["se"] = float((n * np.var(loo_i)) ** 0.5)
    full["p_loo"] = float(lppd_p[trial_pattern].sum() - elpd)
    full["n_data_points"] = n
    full["loo_i"] = xr.DataArray(loo_i, dims=[TRIAL_DIM])
    full["pareto_k"] = xr.DataArray(k, dims=[TRIAL_DIM])
    full["warning"] = bool(np.any(~(k <= good_k)))

    bad = ~exact & ~(k <= good_k)
    finite_nonexact = np.isfinite(k) & ~exact
    n_bad = int(bad.sum())
    frac_bad = n_bad / n if n else 0.0
    return LooDiagnostics(
        elpd_loo=elpd,
        loo=full,
        n_points=n,
        n_exact=int(exact.sum()),
        n_bad_k=n_bad,
        frac_bad_k=float(frac_bad),
        max_pareto_k=float(k[finite_nonexact].max()) if finite_nonexact.any() else float("nan"),
        good_k=good_k,
        bad_k_tolerance=float(bad_k_tolerance),
        unreliable=bool(frac_bad > bad_k_tolerance),
    )
