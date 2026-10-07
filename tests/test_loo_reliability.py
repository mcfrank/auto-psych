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


def _idata(
    n_obs=200, n_chains=4, n_draws=250, seed=0, n_exact=0, n_heavy=0, heavy_k=1.5
):
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


# ── Trials constant only to rounding ────────────────────────────────────
#
# A heads/tails-symmetric model's p_left on a mirror pair (HH vs TT) is 0.5 in
# exact arithmetic, but in floating point it comes out 0.5 to within an ulp or
# two, so the trial's log-likelihood takes a few adjacent float values across
# the draws. arviz's PSIS fits a Pareto tail to excesses of 1-3 x 1.1e-16,
# gets a finite k with a NaN scale, and smooths every weight to NaN: the
# trial's loo_i and the total ELPD-LOO are NaN. That dropped 6 of run 2's 8
# carried models at the start of its experiment 2 (October 2026), experiment
# 1's winner among them, and one of run 1's. The column below is one of those
# trials (personal_ideal_with_personal_lapse, run 2, experiment 2, trial 2516),
# value by value.
_ROUNDING_LEVEL_COLUMN = (
    (-0.6931471805599455, 106),
    (-0.6931471805599453, 11865),
    (-0.6931471805599452, 28),
    (-0.6931471805599451, 1),
)


def _idata_with_rounding_level_trial(n_ordinary=49, seed=0):
    """One chain of 12 000 draws: the real near-constant trial first, then
    ``n_ordinary`` well-behaved trials."""
    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(seed)
    column = np.concatenate(
        [np.full(count, value) for value, count in _ROUNDING_LEVEL_COLUMN]
    )
    rng.shuffle(column)
    n_draws = column.shape[0]
    ordinary = -0.6 + rng.normal(0.0, 0.05, size=(n_draws, n_ordinary))
    arr = np.concatenate([column[:, None], ordinary], axis=1)[None, :, :]
    coords = {"chain": [0], "draw": np.arange(n_draws)}
    ll = xr.Dataset(
        {"response": (("chain", "draw", "obs"), arr)},
        coords={**coords, "obs": np.arange(arr.shape[2])},
    )
    post = xr.Dataset(
        {"theta": (("chain", "draw"), rng.normal(size=(1, n_draws)))},
        coords=coords,
    )
    return az.InferenceData(posterior=post, log_likelihood=ll)


def test_the_fixture_reproduces_arvizs_nan():
    import arviz as az

    loo = az.loo(_idata_with_rounding_level_trial(), pointwise=True)
    assert np.isnan(np.asarray(loo.loo_i)[0])
    assert np.isnan(float(loo.elpd_loo))


def test_a_trial_constant_to_rounding_scores_its_constant_value():
    import arviz as az

    idata = _idata_with_rounding_level_trial()
    plain = az.loo(idata, pointwise=True)

    diag = lr.loo_diagnostics(idata)

    loo_i = np.asarray(diag.loo.loo_i)
    assert np.isfinite(diag.elpd_loo)
    assert loo_i[0] == pytest.approx(np.log(0.5), abs=1e-12)
    # The other trials are scored exactly as arviz scores them.
    np.testing.assert_array_equal(loo_i[1:], np.asarray(plain.loo_i)[1:])
    assert diag.elpd_loo == pytest.approx(float(loo_i.sum()))
    assert diag.n_exact == 1
    assert diag.unreliable is False


def test_the_fit_itself_is_left_as_it_was():
    idata = _idata_with_rounding_level_trial()
    before = np.array(idata.log_likelihood["response"].values)

    lr.loo_diagnostics(idata)

    np.testing.assert_array_equal(idata.log_likelihood["response"].values, before)


def test_a_comparison_of_such_fits_is_finite():
    import arviz as az

    a = lr.loo_diagnostics(_idata_with_rounding_level_trial(seed=1))
    b = lr.loo_diagnostics(_idata_with_rounding_level_trial(seed=2))

    table = az.compare({"a": a.loo, "b": b.loo}, ic="loo")

    assert np.isfinite(table["elpd_loo"].to_numpy()).all()
    assert np.isfinite(table["dse"].to_numpy()).all()
