"""A candidate's admission fit is stopped at a wall-clock limit.

``sample_fits_time_limited`` runs each sampling run in its own process
session and, at the limit, kills the whole group (the fit and the chain
processes PyMC forks) — the sampling stops, not just the wait for it — and
removes a half-written fit file. A model's own failure is reported, a process
that dies without reporting is the harness's failure, and a failed run is not
sampled again. ``fit_model(time_limit_sec=...)`` applies the limit to the
first fit and to a near-miss refit alike.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.models import pymc_inference as pi
from tests import fit_process_stand_ins as stand_ins


@pytest.fixture(autouse=True)
def _clean_fit_cache():
    pi.clear_fit_cache()
    yield
    pi.clear_fit_cache()


def _request(tmp_path, name="m"):
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / f"{name}.py").write_text(f"# stub model {name}\n", encoding="utf-8")
    responses = tmp_path / "responses.csv"
    responses.write_text("chose_left\n1\n", encoding="utf-8")
    settings = pi.resolve_fit_settings(name, models_dir, {"target_accept": 0.8})
    return pi.TimeLimitedFit(name, models_dir, responses, settings, tmp_path / "cache")


def _running(pid: int) -> bool:
    """A process that exists and is not a zombie."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


def test_a_fit_still_sampling_at_its_limit_is_stopped_with_its_chains(tmp_path):
    request = _request(tmp_path)
    start = time.monotonic()
    [outcome] = pi.sample_fits_time_limited(
        [request], time_limit_sec=3, _target=stand_ins.sample_with_a_chain_forever
    )
    assert time.monotonic() - start < 30
    assert isinstance(outcome, pi.FitTimeLimitExceeded)
    assert "0.05-minute limit" in str(outcome) and "target_accept 0.8" in str(outcome)
    fit_pid, chain_pid = map(int, (request.cache_dir / "pids").read_text().split())
    deadline = time.monotonic() + 10
    while (_running(fit_pid) or _running(chain_pid)) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _running(fit_pid) and not _running(chain_pid)
    assert not list(request.cache_dir.glob("*.partial"))
    assert not request.nc_path().exists()


def test_a_run_that_failed_is_not_sampled_again(tmp_path):
    request = _request(tmp_path)
    [first] = pi.sample_fits_time_limited(
        [request], time_limit_sec=60, _target=stand_ins.fail_as_the_model
    )
    [again] = pi.sample_fits_time_limited(
        [request], time_limit_sec=60, _target=stand_ins.die_without_reporting
    )
    assert again is first


def test_a_run_that_timed_out_stays_timed_out_even_if_it_left_a_fit(tmp_path):
    """Killed at its limit just after writing its fit: still a timeout, for
    every later caller (a sequential admission would have seen the timeout)."""
    request = _request(tmp_path)
    [outcome] = pi.sample_fits_time_limited(
        [request], time_limit_sec=3, _target=stand_ins.write_the_fit_and_hang
    )
    assert isinstance(outcome, pi.FitTimeLimitExceeded)
    assert request.nc_path().exists()
    with pytest.raises(pi.FitTimeLimitExceeded):
        pi.fit_model(
            request.name,
            request.models_dir,
            request.responses_path,
            cache_dir=request.cache_dir,
            target_accept=0.8,
            time_limit_sec=3,
        )


def test_the_models_own_failure_is_reported_by_name(tmp_path):
    [outcome] = pi.sample_fits_time_limited(
        [_request(tmp_path)], time_limit_sec=60, _target=stand_ins.fail_as_the_model
    )
    assert isinstance(outcome, pi.FitWorkerFailure)
    assert str(outcome) == "ValueError: the model's own failure"


def test_a_process_that_dies_without_reporting_is_an_infrastructure_failure(tmp_path):
    with pytest.raises(pi.FitInfrastructureFailure, match="exited with code 3"):
        pi.sample_fits_time_limited(
            [_request(tmp_path)],
            time_limit_sec=60,
            _target=stand_ins.die_without_reporting,
        )


def test_fits_run_concurrently_and_report_in_request_order(tmp_path):
    requests = [_request(tmp_path, name) for name in ("a", "b", "c")]
    start = time.monotonic()
    outcomes = pi.sample_fits_time_limited(
        requests, time_limit_sec=60, workers=3, _target=stand_ins.write_the_fit
    )
    assert outcomes == [None, None, None]
    assert all(r.nc_path().exists() for r in requests)
    assert time.monotonic() - start < 60


def test_a_fit_already_on_disk_is_not_sampled(tmp_path):
    request = _request(tmp_path)
    request.cache_dir.mkdir()
    request.nc_path().write_text("fit")
    [outcome] = pi.sample_fits_time_limited(
        [request], time_limit_sec=60, _target=stand_ins.die_without_reporting
    )
    assert outcome is None


def test_the_limit_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        pi.sample_fits_time_limited([_request(tmp_path)], time_limit_sec=0)


# ---------------------------------------------------------------------------
# fit_model(time_limit_sec=...)
# ---------------------------------------------------------------------------


class _StubFit(dict):
    fingerprint = "first-fit-fingerprint"


def _limited_fit_model(monkeypatch, *, outcomes, near_miss):
    """fit_model with the child process and the loading stubbed."""
    runs = []

    def fake_sample(requests, *, time_limit_sec):
        [request] = requests
        if request.nc_path().exists():
            return [None]  # on disk: nothing to sample
        runs.append((request.settings["target_accept"], time_limit_sec))
        outcome = outcomes.pop(0)
        if outcome is None:
            request.cache_dir.mkdir(parents=True, exist_ok=True)
            request.nc_path().write_text("fit")
        return [outcome]

    def fake_fit_once(name, models_dir, responses_path, settings, cache_dir):
        return _StubFit(target_accept=settings["target_accept"])

    monkeypatch.setattr(pi, "sample_fits_time_limited", fake_sample)
    monkeypatch.setattr(pi, "_fit_once", fake_fit_once)
    monkeypatch.setattr(
        pi,
        "convergence_problems_of",
        lambda f: ["R-hat 1.1"] if f["target_accept"] < 0.95 else [],
    )
    monkeypatch.setattr(
        pi,
        "convergence_diagnostics_of",
        lambda f: pi.ConvergenceDiagnostics(0, 4000, 1.1 if near_miss else 2.5, 50),
    )
    return runs


def test_the_limit_applies_to_the_first_fit_and_to_a_near_miss_refit(
    tmp_path, monkeypatch
):
    request = _request(tmp_path)
    runs = _limited_fit_model(monkeypatch, outcomes=[None, None], near_miss=True)
    fitted = pi.fit_model(
        "m",
        request.models_dir,
        request.responses_path,
        cache_dir=request.cache_dir,
        target_accept=0.8,
        chains=4,
        time_limit_sec=900,
    )
    assert runs == [(0.8, 900), (0.95, 900)]
    assert fitted["target_accept"] == 0.95


def test_a_timed_out_refit_raises_the_time_limit(tmp_path, monkeypatch):
    request = _request(tmp_path)
    timeout = pi.FitTimeLimitExceeded("m", 900, 0.95)
    _limited_fit_model(monkeypatch, outcomes=[None, timeout], near_miss=True)
    with pytest.raises(pi.FitTimeLimitExceeded):
        pi.fit_model(
            "m",
            request.models_dir,
            request.responses_path,
            cache_dir=request.cache_dir,
            target_accept=0.8,
            chains=4,
            time_limit_sec=900,
        )


def test_a_cached_fit_is_loaded_without_a_child_process(tmp_path, monkeypatch):
    request = _request(tmp_path)
    runs = _limited_fit_model(monkeypatch, outcomes=[], near_miss=False)
    request.cache_dir.mkdir()
    request.nc_path().write_text("fit")
    fitted = pi.fit_model(
        "m",
        request.models_dir,
        request.responses_path,
        cache_dir=request.cache_dir,
        target_accept=0.8,
        chains=4,
        time_limit_sec=900,
    )
    assert runs == [] and fitted["target_accept"] == 0.8


# ---------------------------------------------------------------------------
# Real sampling
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_time_limited_fit_is_the_fit_an_unlimited_one_makes(tmp_path):
    from tests.paths import PYMC_MODEL_FIXTURES_DIR

    responses = PYMC_MODEL_FIXTURES_DIR / "responses.csv"
    settings = {"draws": 200, "tune": 200, "chains": 2, "cores": 2}
    limited = pi.fit_model(
        "representativeness",
        PYMC_MODEL_FIXTURES_DIR,
        responses,
        cache_dir=tmp_path / "limited",
        time_limit_sec=600,
        **settings,
    )
    unlimited = pi.fit_model(
        "representativeness",
        PYMC_MODEL_FIXTURES_DIR,
        responses,
        cache_dir=tmp_path / "unlimited",
        **settings,
    )
    assert limited.fingerprint == unlimited.fingerprint
    assert limited.elpd_loo() == unlimited.elpd_loo()
    in_memory = pi.fit_model(
        "representativeness",
        PYMC_MODEL_FIXTURES_DIR,
        responses,
        time_limit_sec=600,
        **settings,
    )
    assert in_memory.elpd_loo() == unlimited.elpd_loo()


@pytest.mark.slow
def test_a_real_fit_over_its_limit_is_stopped(tmp_path):
    from tests.paths import PYMC_MODEL_FIXTURES_DIR

    start = time.monotonic()
    with pytest.raises(pi.FitTimeLimitExceeded):
        pi.fit_model(
            "representativeness",
            PYMC_MODEL_FIXTURES_DIR,
            PYMC_MODEL_FIXTURES_DIR / "responses.csv",
            cache_dir=tmp_path,
            draws=10**6,
            tune=1000,
            chains=2,
            cores=2,
            time_limit_sec=20,
        )
    assert time.monotonic() - start < 60
    assert not list(tmp_path.glob("*.nc")) and not list(tmp_path.glob(".*partial"))
