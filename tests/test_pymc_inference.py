"""Unit tests for src/models/pymc_inference.py — fast, no MCMC."""

from __future__ import annotations

import csv
import hashlib

import numpy as np
import pytest

from src.models import pymc_inference as pi
from tests.paths import PYMC_MODEL_FIXTURES_DIR


def test_load_pymc_model_returns_pm_model():
    import pymc as pm

    model = pi.load_pymc_model("bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR)
    assert isinstance(model, pm.Model)


def test_load_pymc_model_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        pi.load_pymc_model("does_not_exist", PYMC_MODEL_FIXTURES_DIR)


def test_pm_data_inputs_lists_all_data_containers():
    model = pi.load_pymc_model("bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR)
    names = pi.pm_data_inputs(model)
    assert set(names) == {"n_a", "h_a", "n_b", "h_b", "chose_left"}


def test_observed_response_data_identifies_y_via_graph():
    model = pi.load_pymc_model("bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR)
    assert pi.observed_response_data(model) == "chose_left"


def test_observed_response_data_works_for_second_fixture():
    model = pi.load_pymc_model("representativeness", PYMC_MODEL_FIXTURES_DIR)
    assert pi.observed_response_data(model) == "chose_left"


def test_extract_observed_pulls_columns_by_name_and_dtype(tmp_path):
    model = pi.load_pymc_model("bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR)
    csv_path = tmp_path / "responses.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_a", "h_a", "n_b", "h_b", "chose_left"])
        w.writeheader()
        w.writerow(
            {"n_a": "10", "h_a": "5", "n_b": "10", "h_b": "5", "chose_left": "1"}
        )
        w.writerow({"n_a": "12", "h_a": "8", "n_b": "8", "h_b": "4", "chose_left": "0"})

    observed = pi.extract_observed(csv_path, model)
    assert set(observed.keys()) == {"n_a", "h_a", "n_b", "h_b", "chose_left"}
    assert np.issubdtype(observed["n_a"].dtype, np.integer)
    assert observed["n_a"].tolist() == [10, 12]
    assert observed["chose_left"].tolist() == [1, 0]


def test_extract_observed_missing_column_raises(tmp_path):
    model = pi.load_pymc_model("bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR)
    csv_path = tmp_path / "responses.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_a", "chose_left"])  # missing h_a, n_b, h_b
        w.writeheader()
        w.writerow({"n_a": "10", "chose_left": "1"})
    with pytest.raises(ValueError, match="missing columns"):
        pi.extract_observed(csv_path, model)


def test_observed_response_data_zero_obs_rvs_raises():
    """A model with no observed RVs should fail loudly."""
    import pymc as pm

    with pm.Model() as bad:
        pm.Data("x", np.zeros(1, dtype="int64"))
        pm.Normal("z", mu=0, sigma=1)  # not observed
    with pytest.raises(ValueError, match="no observed RVs"):
        pi.observed_response_data(bad)


def test_observed_response_data_two_obs_rvs_raises():
    import pymc as pm

    with pm.Model() as bad:
        y1 = pm.Data("y1", np.zeros(1, dtype="int64"))
        y2 = pm.Data("y2", np.zeros(1, dtype="int64"))
        pm.Bernoulli("r1", p=0.5, observed=y1)
        pm.Bernoulli("r2", p=0.5, observed=y2)
    with pytest.raises(ValueError, match="expected exactly one"):
        pi.observed_response_data(bad)


def test_cache_key_changes_when_model_or_data_changes(tmp_path):
    """Two different responses CSVs must produce different cache keys."""
    csv1 = tmp_path / "a.csv"
    csv2 = tmp_path / "b.csv"
    csv1.write_text("col,col2\n1,2\n")
    csv2.write_text("col,col2\n3,4\n")
    k1 = pi._cache_key("bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR, csv1)
    k2 = pi._cache_key("bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR, csv2)
    assert k1 != k2


def test_cache_key_changes_when_sampler_settings_change(tmp_path):
    """A fit sampled under different MCMC settings must not be reused — the cache
    key (and the on-disk .nc fingerprint, which uses the same signature) must
    distinguish draws/tune/chains/cores/seed."""
    csv = tmp_path / "a.csv"
    csv.write_text("col,col2\n1,2\n")
    base = pi._cache_key("bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR, csv, {"draws": 500})
    more = pi._cache_key("bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR, csv, {"draws": 2000})
    assert base != more
    # Default settings and an explicit-but-equal spec must collide (cache hit).
    default = pi._cache_key("bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR, csv)
    explicit = pi._cache_key(
        "bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR, csv, dict(pi._FIT_DEFAULTS)
    )
    assert default == explicit


class _StopSampling(Exception):
    """Raised by the fake sampler once it has captured pm.sample's kwargs."""


def _write_fair_coin_responses(path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_a", "h_a", "n_b", "h_b", "chose_left"])
        w.writeheader()
        w.writerow({"n_a": "10", "h_a": "5", "n_b": "10", "h_b": "5", "chose_left": "1"})
        w.writerow({"n_a": "12", "h_a": "8", "n_b": "8", "h_b": "4", "chose_left": "0"})


def test_fit_model_funnels_centralized_production_settings_into_pm_sample(
    tmp_path, monkeypatch
):
    """Every fit routes the centralized production sampler config into pm.sample,
    including an EXPLICIT target_accept — so NUTS no longer silently uses PyMC's
    0.8 default, and 'what settings did this run use?' has one answer."""
    import pymc as pm

    from src.models import mcmc_defaults as md

    captured = {}

    def fake_sample(*args, **kwargs):
        captured.update(kwargs)
        raise _StopSampling

    monkeypatch.setattr(pm, "sample", fake_sample)

    csv_path = tmp_path / "responses.csv"
    _write_fair_coin_responses(csv_path)

    with pytest.raises(_StopSampling):
        pi.fit_model("bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR, csv_path)

    assert captured["draws"] == md.PRODUCTION_DRAWS
    assert captured["tune"] == md.PRODUCTION_TUNE
    assert captured["chains"] == md.PRODUCTION_CHAINS
    assert captured["target_accept"] == md.PRODUCTION_TARGET_ACCEPT


def test_fit_defaults_derive_from_centralized_production_config():
    """_FIT_DEFAULTS must not re-declare the production sampler literals — it
    derives them from src.models.mcmc_defaults so the CLI defaults and the
    fit-model fallback can never drift apart."""
    from src.models import mcmc_defaults as md

    assert pi._FIT_DEFAULTS["draws"] == md.PRODUCTION_DRAWS
    assert pi._FIT_DEFAULTS["tune"] == md.PRODUCTION_TUNE
    assert pi._FIT_DEFAULTS["chains"] == md.PRODUCTION_CHAINS
    assert pi._FIT_DEFAULTS["target_accept"] == md.PRODUCTION_TARGET_ACCEPT


def test_sampler_signature_distinguishes_target_accept():
    """target_accept is part of the resolved sampler settings, so a fit sampled
    at 0.8 must not be reused for a request at 0.99 (cache correctness)."""
    sig = pi._sampler_signature({})
    assert "target_accept=" in sig
    assert pi._sampler_signature({"target_accept": 0.8}) != pi._sampler_signature(
        {"target_accept": 0.99}
    )


def test_fit_model_threads_max_treedepth_into_pm_sample(tmp_path, monkeypatch):
    """max_treedepth caps the NUTS trajectory length so a pathologically stiff
    model can't hang a fit; it must reach pm.sample and be folded into the cache
    signature (a capped fit must not be reused for an uncapped request). Defaults
    to PyMC's 10 (uncapped) unless a caller sets it."""
    import pymc as pm

    captured = {}

    def fake_sample(*args, **kwargs):
        captured.update(kwargs)
        raise _StopSampling

    monkeypatch.setattr(pm, "sample", fake_sample)
    csv_path = tmp_path / "responses.csv"
    _write_fair_coin_responses(csv_path)

    with pytest.raises(_StopSampling):
        pi.fit_model(
            "bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR, csv_path, max_treedepth=6
        )

    assert captured["max_treedepth"] == 6
    assert pi._FIT_DEFAULTS["max_treedepth"] == 10  # uncapped by default
    assert pi._sampler_signature({"max_treedepth": 6}) != pi._sampler_signature(
        {"max_treedepth": 10}
    )


def test_thin_posterior_subsamples_to_at_most_max_draws():
    import arviz as az

    idata = az.from_dict(posterior={"x": np.zeros((4, 2000))})  # 8000 total
    thinned = pi._thin_posterior(idata, 500)
    assert thinned.posterior.sizes["chain"] == 4
    assert thinned.posterior.sizes["draw"] == 125  # 500 // 4 per chain
    total = thinned.posterior.sizes["chain"] * thinned.posterior.sizes["draw"]
    assert total <= 500


def test_thin_posterior_is_noop_when_already_within_budget():
    import arviz as az

    idata = az.from_dict(posterior={"x": np.zeros((2, 50))})  # 100 total
    thinned = pi._thin_posterior(idata, 500)
    assert thinned is idata


def test_prior_predict_p_left_returns_per_model_means():
    feature_row = {"n_a": 10, "h_a": 5, "n_b": 10, "h_b": 5, "chose_left": 0}
    pi.clear_model_cache()
    preds = pi.prior_predict_p_left(
        ["bayesian_fair_coin", "representativeness"],
        PYMC_MODEL_FIXTURES_DIR,
        feature_row,
        n_samples=50,
    )
    assert set(preds.keys()) == {"bayesian_fair_coin", "representativeness"}
    # Balanced stimulus → both models near 0.5 under their priors.
    for v in preds.values():
        assert 0.0 < v < 1.0
        assert abs(v - 0.5) < 0.2


def test_expected_information_gain_prior_pymc_nonneg():
    feature_row = {"n_a": 10, "h_a": 7, "n_b": 10, "h_b": 3, "chose_left": 0}
    pi.clear_model_cache()
    eig = pi.expected_information_gain_prior_pymc(
        feature_row,
        ["bayesian_fair_coin", "representativeness"],
        PYMC_MODEL_FIXTURES_DIR,
        n_samples=50,
    )
    assert eig >= 0.0
    assert eig <= 1.0  # EIG over 2 models is at most log2(2) = 1 bit


def _binary_entropy(p: float) -> float:
    import math

    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def test_eig_from_prior_means_matches_hand_computed():
    # Uniform prior over two models with p_left 0.8 / 0.2: marginal p_left is
    # 0.5, H(M) = 1 bit, and P(M|R) is (0.8, 0.2) for either response, so
    # EIG = 1 - H_b(0.8).
    eig = pi.eig_from_prior_means({"m1": 0.8, "m2": 0.2})
    assert eig == pytest.approx(1.0 - _binary_entropy(0.8))


def test_eig_from_prior_means_zero_when_models_agree():
    # Identical predictions: the response carries no information about M.
    assert pi.eig_from_prior_means({"m1": 0.7, "m2": 0.7}) == pytest.approx(0.0)


def test_eig_from_prior_means_weighted_matches_definition():
    import math

    preds = {"m1": 0.8, "m2": 0.2}
    weights = {"m1": 3.0, "m2": 1.0}
    p_model = {"m1": 0.75, "m2": 0.25}
    p_left = sum(preds[m] * p_model[m] for m in preds)
    h_m = -sum(p * math.log2(p) for p in p_model.values())
    h_given = 0.0
    for response_p, lik in ((p_left, preds), (1.0 - p_left, {m: 1.0 - preds[m] for m in preds})):
        post = {m: lik[m] * p_model[m] / response_p for m in preds}
        h_given += response_p * -sum(p * math.log2(p) for p in post.values() if p > 0)
    expected = h_m - h_given

    eig = pi.eig_from_prior_means(preds, model_weights=weights)
    assert eig == pytest.approx(expected)


def test_eig_from_prior_means_degenerate_p_left_is_zero():
    # All models certain of the same response: marginal p_left hits 1.0.
    assert pi.eig_from_prior_means({"m1": 1.0, "m2": 1.0}) == 0.0


@pytest.mark.parametrize("prediction", [-0.1, 1.1, float("nan"), float("inf")])
def test_eig_from_prior_means_rejects_invalid_probability(prediction):
    with pytest.raises(ValueError, match="m1"):
        pi.eig_from_prior_means({"m1": prediction, "m2": 0.5})


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_eig_from_prior_means_rejects_invalid_weight(weight):
    with pytest.raises(ValueError, match="m1"):
        pi.eig_from_prior_means(
            {"m1": 0.8, "m2": 0.2}, {"m1": weight, "m2": 1.0}
        )


@pytest.mark.slow
def test_fitted_model_predict_p_left_draws_shape_and_mean_consistency(tmp_path):
    """Posterior-predictive per-draw p_left: shape (n_draws, n_stim), values in
    [0, 1], thinning respected, and its mean is exactly predict_p_left."""
    model = pi.load_pymc_model("bayesian_fair_coin", PYMC_MODEL_FIXTURES_DIR)
    fitted = pi.fit_model(
        "bayesian_fair_coin",
        PYMC_MODEL_FIXTURES_DIR,
        PYMC_MODEL_FIXTURES_DIR / "responses.csv",
        cache_dir=tmp_path,
        draws=100,
        tune=100,
        chains=2,
    )
    rows = [
        {"n_a": 10, "h_a": 5, "n_b": 10, "h_b": 5, "chose_left": 0},
        {"n_a": 8, "h_a": 7, "n_b": 6, "h_b": 3, "chose_left": 0},
        {"n_a": 4, "h_a": 0, "n_b": 4, "h_b": 2, "chose_left": 0},
    ]
    stim_data = pi.make_stim_data(model, rows)

    draws = fitted.predict_p_left_draws(stim_data, seed=7)
    assert draws.shape == (200, len(rows))  # 2 chains x 100 draws
    assert np.all((0.0 <= draws) & (draws <= 1.0))

    thinned = fitted.predict_p_left_draws(stim_data, seed=7, max_draws=50)
    assert thinned.shape[0] <= 50
    assert thinned.shape[1] == len(rows)

    means = fitted.predict_p_left(stim_data, seed=7)
    np.testing.assert_allclose(draws.mean(axis=0), means)


def test_prior_predict_p_left_draws_shape_and_mean_consistency():
    rows = [
        {"n_a": 10, "h_a": 5, "n_b": 10, "h_b": 5, "chose_left": 0},
        {"n_a": 8, "h_a": 7, "n_b": 6, "h_b": 3, "chose_left": 0},
    ]
    names = ["bayesian_fair_coin", "representativeness"]
    pi.clear_model_cache()
    draws = pi.prior_predict_p_left_draws(names, PYMC_MODEL_FIXTURES_DIR, rows, n_samples=40, seed=3)
    batch = pi.prior_predict_p_left_batch(names, PYMC_MODEL_FIXTURES_DIR, rows, n_samples=40, seed=3)
    for name in names:
        assert draws[name].shape == (40, len(rows))
        assert np.all((0.0 <= draws[name]) & (draws[name] <= 1.0))
        # The batched means are exactly the per-draw means (same draws).
        np.testing.assert_allclose(draws[name].mean(axis=0), batch[name])


def test_prior_predict_p_left_batch_matches_per_row():
    rows = [
        {"n_a": 10, "h_a": 5, "n_b": 10, "h_b": 5, "chose_left": 0},
        {"n_a": 8, "h_a": 7, "n_b": 6, "h_b": 3, "chose_left": 0},
        {"n_a": 4, "h_a": 0, "n_b": 4, "h_b": 2, "chose_left": 0},
    ]
    names = ["bayesian_fair_coin", "representativeness"]
    pi.clear_model_cache()
    batch = pi.prior_predict_p_left_batch(
        names, PYMC_MODEL_FIXTURES_DIR, rows, n_samples=50, seed=11
    )
    assert set(batch.keys()) == set(names)
    for name in names:
        assert batch[name].shape == (len(rows),)
    # Same seed → same prior parameter draws → the batched means must match the
    # per-row path (which re-seeds identically for every row).
    for i, row in enumerate(rows):
        per_row = pi.prior_predict_p_left(names, PYMC_MODEL_FIXTURES_DIR, row, n_samples=50, seed=11)
        for name in names:
            assert batch[name][i] == pytest.approx(per_row[name], abs=1e-8)


# ── Sampling diagnostics must never fabricate a clean result ────────────────
#
# `_warn_sampling_diagnostics` used to read a missing `diverging` stat as
# "0 divergences" and an un-computable R-hat as NaN, i.e. exactly the two
# values that mean "this fit is fine". A fit whose quality could not be checked
# must say so instead of impersonating a healthy one.


class _FakeDataArray:
    """Minimal stand-in for the xarray DataArray in ``idata.sample_stats``."""

    def __init__(self, values):
        self.values = values


class _FakeSampleStats:
    def __init__(self, values: dict):
        self._values = values

    def __contains__(self, key):
        return key in self._values

    def __getitem__(self, key):
        return self._values[key]


class _FakeIdata:
    def __init__(self, sample_stats=None):
        if sample_stats is not None:
            self.sample_stats = sample_stats


def test_diagnostics_report_that_divergences_could_not_be_checked(capsys, monkeypatch):
    monkeypatch.setattr(pi, "_max_rhat", lambda idata: 1.0)
    pi._warn_sampling_diagnostics("m", _FakeIdata())
    err = capsys.readouterr().err
    assert "divergence" in err.lower()
    assert "could not" in err.lower() or "no divergence statistic" in err.lower()


def test_diagnostics_report_an_unavailable_rhat(capsys, monkeypatch):
    monkeypatch.setattr(pi, "_max_rhat", lambda idata: float("nan"))
    stats = _FakeSampleStats({"diverging": _FakeDataArray(np.zeros(10))})
    pi._warn_sampling_diagnostics("m", _FakeIdata(stats))
    err = capsys.readouterr().err
    assert "r-hat" in err.lower()


def test_diagnostics_stay_quiet_for_a_healthy_fit(capsys, monkeypatch):
    monkeypatch.setattr(pi, "_max_rhat", lambda idata: 1.001)
    stats = _FakeSampleStats({"diverging": _FakeDataArray(np.zeros(10))})
    pi._warn_sampling_diagnostics("m", _FakeIdata(stats))
    assert capsys.readouterr().err == ""


def test_diagnostics_propagate_unexpected_errors(monkeypatch):
    class _Exploding:
        def __contains__(self, key):
            return True

        def __getitem__(self, key):
            raise TypeError("sample_stats is not subscriptable like that")

    monkeypatch.setattr(pi, "_max_rhat", lambda idata: 1.0)
    with pytest.raises(TypeError):
        pi._warn_sampling_diagnostics("m", _FakeIdata(_Exploding()))


def test_in_process_cache_hits_rerun_sampling_diagnostics(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m.py").write_text("# model\n", encoding="utf-8")
    responses = tmp_path / "responses.csv"
    responses.write_text("chose_left\n1\n", encoding="utf-8")
    idata = object()
    cached = pi.FittedModel(name="m", model=object(), idata=idata, fingerprint="fp")
    key = pi._cache_key("m", models_dir, responses, {})
    pi._FIT_CACHE[key] = cached
    checked = []
    monkeypatch.setattr(
        pi, "_warn_sampling_diagnostics", lambda name, value: checked.append((name, value))
    )

    result = pi.fit_models_cached(["m"], models_dir, responses)

    assert result == {"m": cached}
    assert checked == [("m", idata)]
    pi.clear_fit_cache()


def test_disk_cache_hits_rerun_sampling_diagnostics(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "m.py"
    model_path.write_text("# model\n", encoding="utf-8")
    responses = tmp_path / "responses.csv"
    responses.write_text("chose_left\n1\n", encoding="utf-8")
    signature = pi._sampler_signature(pi._FIT_DEFAULTS)
    fingerprint = hashlib.sha256(
        (pi._sha256_file(model_path) + pi._sha256_file(responses) + signature).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / f"m.{fingerprint}.nc").touch()
    idata = object()

    class _FakeArviz:
        @staticmethod
        def from_netcdf(path):
            return idata

    monkeypatch.setattr(pi, "_import_pymc", lambda: object())
    monkeypatch.setattr(pi, "_import_arviz", lambda: _FakeArviz())
    monkeypatch.setattr(pi, "load_pymc_model", lambda name, directory: object())
    checked = []
    monkeypatch.setattr(
        pi, "_warn_sampling_diagnostics", lambda name, value: checked.append((name, value))
    )

    fitted = pi.fit_model("m", models_dir, responses, cache_dir=cache_dir)

    assert fitted.idata is idata
    assert checked == [("m", idata)]
