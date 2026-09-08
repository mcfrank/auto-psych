"""Fast tests for the PSIS-LOO reliability diagnostic.

arviz flags a whole LOO estimate as unreliable when *any* observation's Pareto
k exceeds ``good_k``. Two things make that verdict wrong for our Bernoulli
choice models: (1) a trial whose log-likelihood is identical across every draw
(a clipped, saturated ``p_left``) has no importance-weight tail, arviz reports
k = inf, yet the LOO term there is exact; (2) one influential trial in
thousands cannot bias a total ELPD by more than a few nats. The diagnostic here
exempts exact trials and judges the rest by the *proportion* with high k.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models import loo_reliability as lr


def _idata(n_obs=200, n_chains=4, n_draws=250, seed=0, n_exact=0, n_heavy=0, heavy_k=1.5):
    """Log-likelihood draws: well-behaved, plus ``n_exact`` constant trials and
    ``n_heavy`` trials with Pareto-tailed importance weights (GPD shape ≈ heavy_k)."""
    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(seed)
    arr = -0.5 + rng.normal(0.0, 0.02, size=(n_chains, n_draws, n_obs))
    arr[:, :, :n_exact] = np.log(1e-6)
    arr[:, :, n_exact : n_exact + n_heavy] = -0.5 - heavy_k * rng.exponential(
        1.0, size=(n_chains, n_draws, n_heavy)
    )
    coords = {"chain": np.arange(n_chains), "draw": np.arange(n_draws)}
    ll = xr.Dataset(
        {"response": (("chain", "draw", "obs"), arr)},
        coords={**coords, "obs": np.arange(n_obs)},
    )
    post = xr.Dataset(
        {"theta": (("chain", "draw"), rng.normal(size=(n_chains, n_draws)))},
        coords=coords,
    )
    return az.InferenceData(posterior=post, log_likelihood=ll)


def test_exact_trials_are_exempt_and_counted():
    """Constant-log-likelihood trials give arviz k = inf (its ``warning`` is
    True) but the diagnostic counts them as exact, not bad."""
    import arviz as az

    idata = _idata(n_exact=20)
    assert bool(az.loo(idata, pointwise=True).warning) is True

    diag = lr.loo_diagnostics(idata)

    assert diag.n_points == 200
    assert diag.n_exact == 20
    assert diag.n_bad_k == 0
    assert diag.frac_bad_k == pytest.approx(0.0)
    assert diag.unreliable is False


def test_heavy_tailed_trials_above_tolerance_are_unreliable():
    idata = _idata(n_heavy=20)  # 10 % of trials, tolerance is 1 %

    diag = lr.loo_diagnostics(idata)

    assert diag.n_bad_k == 20
    assert diag.frac_bad_k == pytest.approx(0.1)
    assert np.isfinite(diag.max_pareto_k) and diag.max_pareto_k > diag.good_k
    assert diag.unreliable is True


def test_heavy_tailed_trials_below_tolerance_stay_reliable():
    idata = _idata(n_heavy=1)  # 0.5 % of trials < 1 % tolerance

    diag = lr.loo_diagnostics(idata)

    assert diag.n_bad_k == 1
    assert diag.unreliable is False
    # The same fit IS unreliable under a stricter tolerance.
    assert lr.loo_diagnostics(idata, bad_k_tolerance=0.0).unreliable is True


def test_exact_and_heavy_trials_are_distinguished():
    idata = _idata(n_exact=30, n_heavy=4)  # 2 % bad among all 200 trials

    diag = lr.loo_diagnostics(idata)

    assert diag.n_exact == 30
    assert diag.n_bad_k == 4
    assert diag.frac_bad_k == pytest.approx(4 / 200)
    assert diag.unreliable is True


def test_elpd_matches_arviz_and_pointwise_loo_is_kept():
    import arviz as az

    idata = _idata(n_exact=5)
    diag = lr.loo_diagnostics(idata)

    assert diag.elpd_loo == pytest.approx(float(az.loo(idata).elpd_loo))
    # The pointwise ELPDData is kept so az.compare can reuse it (no recompute).
    assert diag.loo.pareto_k.shape == (200,)
    assert diag.good_k == pytest.approx(float(diag.loo.good_k))


def test_tolerance_must_be_a_proportion():
    idata = _idata()
    with pytest.raises(ValueError, match="bad_k_tolerance"):
        lr.loo_diagnostics(idata, bad_k_tolerance=-0.1)
    with pytest.raises(ValueError, match="bad_k_tolerance"):
        lr.loo_diagnostics(idata, bad_k_tolerance=1.5)


def test_requires_exactly_one_log_likelihood_variable():
    import arviz as az
    import xarray as xr

    idata = _idata()
    extra = idata.log_likelihood["response"].rename("other")
    two_vars = az.InferenceData(
        posterior=idata.posterior,
        log_likelihood=xr.merge([idata.log_likelihood, extra.to_dataset()]),
    )
    with pytest.raises(ValueError, match="exactly one"):
        lr.loo_diagnostics(two_vars)


def test_fitted_model_memoizes_diagnostics_and_warns_only_when_unreliable(capsys):
    from src.models import pymc_inference as pi

    fit = pi.FittedModel(
        name="sat", model=object(), idata=_idata(n_exact=20), fingerprint="fp"
    )
    first = fit.loo_diagnostics()
    assert fit.loo_diagnostics() is first
    assert fit.elpd_loo() == pytest.approx(first.elpd_loo)
    assert "[warn]" not in capsys.readouterr().err

    heavy = pi.FittedModel(
        name="heavy", model=object(), idata=_idata(n_heavy=20), fingerprint="fp"
    )
    heavy.elpd_loo()
    err = capsys.readouterr().err
    assert "[warn] heavy" in err
    assert "20 of 200" in err
