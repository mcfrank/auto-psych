"""Parallel model fitting (``fit_models_cached`` / ``fit_models_to_cache``).

The inner loop used to fit models one after another while the Slurm task held
more CPUs than one fit's chains could use. ``fit_models_cached`` now samples
the models that need MCMC in a process pool: each child fits one model with
``fit_model``, pins its BLAS threads to one, and writes the ``.nc`` into the
on-disk cache; the parent then loads every fit from that cache. The number of
concurrent fits defaults so ``workers x cores-per-fit`` never exceeds the CPUs
the process is allowed to run on.

The equivalence test at the bottom is the acceptance test: the same models
fit sequentially and in parallel must produce identical cache fingerprints
and identical ELPD-LOO values. It samples for real, so it is ``slow``.
"""

from __future__ import annotations

import os
from concurrent.futures import CancelledError
from pathlib import Path

import pytest

from src.models import pymc_inference as pi
from tests.paths import PYMC_MODEL_FIXTURES_DIR

FIXTURE_MODELS = ["bayesian_fair_coin", "representativeness"]
FIXTURE_RESPONSES = PYMC_MODEL_FIXTURES_DIR / "responses.csv"


# ---------------------------------------------------------------------------
# Worker count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cpus, per_fit_cpus, expected",
    [
        (16, [4, 4, 4], 4),  # the sbatch allocation: four 4-chain fits at once
        (8, [4, 4], 2),
        (4, [4], 1),  # exactly one fit fills the allocation
        (3, [4], 1),  # never zero, even when one fit oversubscribes
        (16, [2, 4], 4),  # the widest fit sets the budget
        (16, [1, 1], 16),
    ],
)
def test_default_fit_workers_never_oversubscribes_the_allocation(
    cpus, per_fit_cpus, expected
):
    assert pi.default_fit_workers(cpus, per_fit_cpus) == expected


def test_default_fit_workers_rejects_an_empty_fit_list():
    with pytest.raises(ValueError, match="no fits"):
        pi.default_fit_workers(16, [])


def test_allocated_cpus_is_the_scheduler_affinity_not_the_node_size():
    assert pi.allocated_cpus() == len(os.sched_getaffinity(0))


def test_fit_cpus_is_the_smaller_of_cores_and_chains():
    assert pi._fit_cpus({"cores": 4, "chains": 2}) == 2
    assert pi._fit_cpus({"cores": 2, "chains": 4}) == 2
    assert pi._fit_cpus({"cores": 4, "chains": 4}) == 4


# ---------------------------------------------------------------------------
# Orchestration: what goes to the pool, what is loaded in the parent
# ---------------------------------------------------------------------------


def _stub_models(tmp_path, names):
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    for name in names:
        (models_dir / f"{name}.py").write_text(f"# stub model {name}\n", encoding="utf-8")
    responses = tmp_path / "responses.csv"
    responses.write_text("chose_left\n1\n0\n", encoding="utf-8")
    return models_dir, responses


class _FakeArviz:
    """``from_netcdf`` hands back a sentinel naming the file it was given."""

    @staticmethod
    def from_netcdf(path):
        return ("idata", Path(path).name)


def _stub_loading(monkeypatch):
    """Make ``fit_model``'s disk-cache path loadable without PyMC or arviz."""
    monkeypatch.setattr(pi, "_import_pymc", lambda: object())
    monkeypatch.setattr(pi, "_import_arviz", lambda: _FakeArviz())
    monkeypatch.setattr(pi, "load_pymc_model", lambda name, directory: object())
    monkeypatch.setattr(pi, "_warn_sampling_diagnostics", lambda name, idata: None)


def _nc_name(name, models_dir, responses, fit_kwargs):
    settings = pi.resolve_fit_settings(name, models_dir, fit_kwargs)
    return pi.cached_fit_path(
        Path("."), name, pi.fit_fingerprint(name, models_dir, responses, settings)
    ).name


def _fake_pool(calls, failures=None):
    """A stand-in for ``_sample_models_in_pool``: touches each model's ``.nc``
    (or reports the failure configured for it) and records how it was called."""

    def pool(names, models_dir, responses_path, cache_dir, fit_kwargs, *, workers, stop_on_failure):
        calls.append(
            {
                "names": list(names),
                "cache_dir": Path(cache_dir),
                "workers": workers,
                "stop_on_failure": stop_on_failure,
            }
        )
        outcomes = {}
        for name in names:
            exc = (failures or {}).get(name)
            if exc is None:
                (Path(cache_dir) / _nc_name(name, models_dir, responses_path, fit_kwargs)).touch()
            outcomes[name] = exc
        return outcomes

    return pool


def _never_pool(*args, **kwargs):
    raise AssertionError("the process pool must not open for this call")


@pytest.fixture(autouse=True)
def _clean_fit_cache():
    pi.clear_fit_cache()
    yield
    pi.clear_fit_cache()


def test_two_pending_fits_are_sampled_in_the_pool_and_loaded_from_the_cache(
    tmp_path, monkeypatch
):
    models_dir, responses = _stub_models(tmp_path, ["a", "b"])
    _stub_loading(monkeypatch)
    calls = []
    monkeypatch.setattr(pi, "_sample_models_in_pool", _fake_pool(calls))
    cache_dir = tmp_path / "cache"

    fits = pi.fit_models_cached(
        ["a", "b"], models_dir, responses, cache_dir=cache_dir, fit_workers=2
    )

    assert calls == [
        {"names": ["a", "b"], "cache_dir": cache_dir, "workers": 2, "stop_on_failure": True}
    ]
    for name in ["a", "b"]:
        assert fits[name].idata == ("idata", _nc_name(name, models_dir, responses, {}))
        assert pi._FIT_CACHE[pi._cache_key(name, models_dir, responses, {})] is fits[name]


def test_a_single_pending_fit_is_sampled_in_process(tmp_path, monkeypatch):
    models_dir, responses = _stub_models(tmp_path, ["only"])
    monkeypatch.setattr(pi, "_sample_models_in_pool", _never_pool)
    fitted = pi.FittedModel(name="only", model=object(), idata=object(), fingerprint="fp")
    monkeypatch.setattr(pi, "fit_model", lambda name, *a, **k: fitted)

    fits = pi.fit_models_cached(["only"], models_dir, responses, fit_workers=4)

    assert fits == {"only": fitted}


def test_the_default_worker_count_comes_from_the_allocation(tmp_path, monkeypatch):
    models_dir, responses = _stub_models(tmp_path, ["a", "b", "c"])
    _stub_loading(monkeypatch)
    calls = []
    monkeypatch.setattr(pi, "_sample_models_in_pool", _fake_pool(calls))
    monkeypatch.setattr(pi, "allocated_cpus", lambda: 16)

    pi.fit_models_cached(
        ["a", "b", "c"], models_dir, responses, cache_dir=tmp_path / "cache",
        chains=4, cores=4,
    )

    assert [c["workers"] for c in calls] == [4]


def test_an_allocation_that_fits_one_model_samples_sequentially(tmp_path, monkeypatch):
    models_dir, responses = _stub_models(tmp_path, ["a", "b"])
    monkeypatch.setattr(pi, "_sample_models_in_pool", _never_pool)
    monkeypatch.setattr(pi, "allocated_cpus", lambda: 4)
    order = []

    def fake_fit(name, *a, **k):
        order.append(name)
        return pi.FittedModel(name=name, model=object(), idata=object(), fingerprint=name)

    monkeypatch.setattr(pi, "fit_model", fake_fit)

    pi.fit_models_cached(["a", "b"], models_dir, responses, chains=4, cores=4)

    assert order == ["a", "b"]


def test_disk_cache_hits_are_loaded_in_the_parent_not_sampled_again(
    tmp_path, monkeypatch
):
    models_dir, responses = _stub_models(tmp_path, ["a", "b", "c"])
    _stub_loading(monkeypatch)
    calls = []
    monkeypatch.setattr(pi, "_sample_models_in_pool", _fake_pool(calls))
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / _nc_name("b", models_dir, responses, {})).touch()

    fits = pi.fit_models_cached(
        ["a", "b", "c"], models_dir, responses, cache_dir=cache_dir, fit_workers=2
    )

    assert [c["names"] for c in calls] == [["a", "c"]]
    assert set(fits) == {"a", "b", "c"}


def test_in_process_cache_hits_never_reach_the_pool(tmp_path, monkeypatch):
    models_dir, responses = _stub_models(tmp_path, ["hit", "miss"])
    monkeypatch.setattr(pi, "_sample_models_in_pool", _never_pool)
    cached = pi.FittedModel(name="hit", model=object(), idata=object(), fingerprint="fp")
    pi._FIT_CACHE[pi._cache_key("hit", models_dir, responses, {})] = cached
    monkeypatch.setattr(pi, "_warn_sampling_diagnostics", lambda name, idata: None)
    fitted = pi.FittedModel(name="miss", model=object(), idata=object(), fingerprint="fp2")
    monkeypatch.setattr(pi, "fit_model", lambda name, *a, **k: fitted)

    fits = pi.fit_models_cached(["hit", "miss"], models_dir, responses, fit_workers=8)

    assert fits == {"hit": cached, "miss": fitted}


def test_a_failed_child_fit_raises_that_failure_after_caching_the_others(
    tmp_path, monkeypatch
):
    models_dir, responses = _stub_models(tmp_path, ["good", "bad"])
    _stub_loading(monkeypatch)
    calls = []
    monkeypatch.setattr(
        pi, "_sample_models_in_pool",
        _fake_pool(calls, failures={"bad": RuntimeError("bad initial energy")}),
    )

    with pytest.raises(RuntimeError, match="bad initial energy"):
        pi.fit_models_cached(
            ["good", "bad"], models_dir, responses, cache_dir=tmp_path / "cache",
            fit_workers=2,
        )

    assert pi._cache_key("good", models_dir, responses, {}) in pi._FIT_CACHE


def test_fit_models_to_cache_reports_failures_by_name_and_caches_the_rest(
    tmp_path, monkeypatch
):
    models_dir, responses = _stub_models(tmp_path, ["good", "bad"])
    _stub_loading(monkeypatch)
    calls = []
    monkeypatch.setattr(
        pi, "_sample_models_in_pool",
        _fake_pool(calls, failures={"bad": RuntimeError("bad initial energy")}),
    )

    failures = pi.fit_models_to_cache(
        ["good", "bad"], models_dir, responses, cache_dir=tmp_path / "cache",
        fit_workers=2,
    )

    assert failures == {"bad": "RuntimeError: bad initial energy"}
    assert calls[0]["stop_on_failure"] is False
    assert pi._cache_key("good", models_dir, responses, {}) in pi._FIT_CACHE
    assert pi._cache_key("bad", models_dir, responses, {}) not in pi._FIT_CACHE


def test_fit_models_to_cache_reports_a_sequential_failure_too(tmp_path, monkeypatch):
    models_dir, responses = _stub_models(tmp_path, ["good", "bad"])
    monkeypatch.setattr(pi, "_sample_models_in_pool", _never_pool)

    def fake_fit(name, *a, **k):
        if name == "bad":
            raise ValueError("cannot bind responses")
        return pi.FittedModel(name=name, model=object(), idata=object(), fingerprint=name)

    monkeypatch.setattr(pi, "fit_model", fake_fit)

    failures = pi.fit_models_to_cache(["bad", "good"], models_dir, responses, fit_workers=1)

    assert failures == {"bad": "ValueError: cannot bind responses"}
    assert pi._cache_key("good", models_dir, responses, {}) in pi._FIT_CACHE


def test_parallel_fits_without_a_cache_dir_use_a_temporary_transport_dir(
    tmp_path, monkeypatch
):
    models_dir, responses = _stub_models(tmp_path, ["a", "b"])
    _stub_loading(monkeypatch)
    calls = []
    monkeypatch.setattr(pi, "_sample_models_in_pool", _fake_pool(calls))

    fits = pi.fit_models_cached(["a", "b"], models_dir, responses, fit_workers=2)

    transport = calls[0]["cache_dir"]
    assert transport != models_dir and not transport.exists()  # cleaned up
    assert set(fits) == {"a", "b"}
    for name in ["a", "b"]:
        assert pi._cache_key(name, models_dir, responses, {}) in pi._FIT_CACHE


@pytest.mark.parametrize("bad_workers", [0, -1])
def test_fit_models_cached_rejects_nonpositive_fit_workers(tmp_path, bad_workers):
    models_dir, responses = _stub_models(tmp_path, ["a"])
    with pytest.raises(ValueError, match="fit_workers"):
        pi.fit_models_cached(["a"], models_dir, responses, fit_workers=bad_workers)


# ---------------------------------------------------------------------------
# The pool driver, run on threads so a fake fit_model is visible to the worker
# ---------------------------------------------------------------------------


def _thread_executor(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(
        pi, "_fit_executor", lambda workers: ThreadPoolExecutor(max_workers=workers)
    )


def _writing_fit(models_dir, responses, fit_kwargs, failing=()):
    """A fake ``fit_model`` that persists the ``.nc`` like the real one."""

    def fake_fit(name, models_dir_, responses_, *, cache_dir, **kw):
        if name in failing:
            raise RuntimeError(f"{name} diverged")
        nc = Path(cache_dir) / _nc_name(name, models_dir, responses, fit_kwargs)
        nc.touch()
        return pi.FittedModel(
            name=name, model=object(), idata=object(), fingerprint=nc.name.split(".")[1]
        )

    return fake_fit


def test_pool_driver_reports_no_failures_when_every_worker_persists_its_fit(
    tmp_path, monkeypatch
):
    models_dir, responses = _stub_models(tmp_path, ["a", "b", "c"])
    _thread_executor(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setattr(pi, "fit_model", _writing_fit(models_dir, responses, {}))
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    outcomes = pi._sample_models_in_pool(
        ["a", "b", "c"], models_dir, responses, cache_dir, {},
        workers=2, stop_on_failure=True,
    )

    assert outcomes == {"a": None, "b": None, "c": None}
    assert sorted(p.name for p in cache_dir.glob("*.nc")) == sorted(
        _nc_name(n, models_dir, responses, {}) for n in ["a", "b", "c"]
    )


def test_pool_driver_records_a_worker_failure_and_continues_when_asked(
    tmp_path, monkeypatch
):
    models_dir, responses = _stub_models(tmp_path, ["a", "b", "c"])
    _thread_executor(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setattr(
        pi, "fit_model", _writing_fit(models_dir, responses, {}, failing={"b"})
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    outcomes = pi._sample_models_in_pool(
        ["a", "b", "c"], models_dir, responses, cache_dir, {},
        workers=1, stop_on_failure=False,
    )

    assert outcomes["a"] is None and outcomes["c"] is None
    assert isinstance(outcomes["b"], RuntimeError) and "b diverged" in str(outcomes["b"])


def test_pool_driver_stops_submitting_after_a_failure_when_asked(tmp_path, monkeypatch):
    names = [f"m{i}" for i in range(6)]
    models_dir, responses = _stub_models(tmp_path, names)
    _thread_executor(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setattr(
        pi, "fit_model", _writing_fit(models_dir, responses, {}, failing={"m0"})
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    outcomes = pi._sample_models_in_pool(
        names, models_dir, responses, cache_dir, {}, workers=1, stop_on_failure=True
    )

    assert set(outcomes) == set(names)  # every name has an outcome
    assert isinstance(outcomes["m0"], RuntimeError)
    # Whatever had not started by the time m0 failed was cancelled, not fit.
    assert all(
        outcome is None or isinstance(outcome, (RuntimeError, CancelledError))
        for outcome in outcomes.values()
    )


def test_pool_driver_refuses_a_worker_that_wrote_no_fit(tmp_path, monkeypatch):
    models_dir, responses = _stub_models(tmp_path, ["a", "b"])
    _thread_executor(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setattr(
        pi, "fit_model",
        lambda name, *a, **k: pi.FittedModel(
            name=name, model=object(), idata=object(),
            fingerprint=_nc_name(name, models_dir, responses, {}).split(".")[1],
        ),
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with pytest.raises(RuntimeError, match="wrote no fit"):
        pi._sample_models_in_pool(
            ["a", "b"], models_dir, responses, cache_dir, {},
            workers=2, stop_on_failure=True,
        )


def test_pool_driver_refuses_a_fingerprint_the_parent_did_not_expect(
    tmp_path, monkeypatch
):
    models_dir, responses = _stub_models(tmp_path, ["a", "b"])
    _thread_executor(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")

    def fake_fit(name, models_dir_, responses_, *, cache_dir, **kw):
        (Path(cache_dir) / _nc_name(name, models_dir, responses, {})).touch()
        return pi.FittedModel(name=name, model=object(), idata=object(), fingerprint="other")

    monkeypatch.setattr(pi, "fit_model", fake_fit)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with pytest.raises(RuntimeError, match="fingerprint"):
        pi._sample_models_in_pool(
            ["a", "b"], models_dir, responses, cache_dir, {},
            workers=2, stop_on_failure=True,
        )


def test_worker_pins_threads_to_one_and_returns_the_fingerprint(tmp_path, monkeypatch):
    models_dir, responses = _stub_models(tmp_path, ["a"])
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    seen = {}

    def fake_fit(name, models_dir_, responses_, *, cache_dir, **kw):
        seen.update({var: os.environ[var] for var in pi._SINGLE_THREAD_ENV})
        seen["kw"] = kw
        return pi.FittedModel(name=name, model=object(), idata=object(), fingerprint="fp1")

    monkeypatch.setattr(pi, "fit_model", fake_fit)

    fingerprint = pi._fit_model_in_worker(
        "a", models_dir, responses, tmp_path / "cache", {"chains": 2}
    )

    assert fingerprint == "fp1"
    assert seen["kw"] == {"chains": 2}
    assert {seen[var] for var in pi._SINGLE_THREAD_ENV} == {"1"}


def test_real_pool_reports_unloadable_models_as_failures(tmp_path):
    """The spawn pool end to end, without MCMC: two stub files that define no
    ``model`` fail inside the children, and the failures come back by name."""
    models_dir, responses = _stub_models(tmp_path, ["stub_a", "stub_b"])

    failures = pi.fit_models_to_cache(
        ["stub_a", "stub_b"], models_dir, responses, cache_dir=tmp_path / "cache",
        fit_workers=2, chains=1, cores=1,
    )

    assert set(failures) == {"stub_a", "stub_b"}
    for name, reason in failures.items():
        assert reason.startswith("TypeError: ")
        assert f"{name}.py" in reason and "module-level `model" in reason


# ---------------------------------------------------------------------------
# Acceptance: sequential and parallel fits are the same fits
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_parallel_and_sequential_fits_are_identical(tmp_path, monkeypatch):
    """Same models, same data, same sampler settings: the parallel path must
    write the same cache fingerprints and score the same ELPD-LOO as the
    sequential path. Two chains on two cores per fit, two workers."""
    pool_calls = []
    real_pool = pi._sample_models_in_pool

    def spy(*args, **kwargs):
        pool_calls.append(kwargs["workers"])
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(pi, "_sample_models_in_pool", spy)
    settings = {"draws": 200, "tune": 200, "chains": 2, "cores": 2}

    pi.clear_fit_cache()
    sequential = pi.fit_models_cached(
        FIXTURE_MODELS,
        PYMC_MODEL_FIXTURES_DIR,
        FIXTURE_RESPONSES,
        cache_dir=tmp_path / "sequential",
        fit_workers=1,
        **settings,
    )
    assert pool_calls == []

    pi.clear_fit_cache()
    parallel = pi.fit_models_cached(
        FIXTURE_MODELS,
        PYMC_MODEL_FIXTURES_DIR,
        FIXTURE_RESPONSES,
        cache_dir=tmp_path / "parallel",
        fit_workers=2,
        **settings,
    )
    assert pool_calls == [2]

    sequential_files = sorted(p.name for p in (tmp_path / "sequential").glob("*.nc"))
    parallel_files = sorted(p.name for p in (tmp_path / "parallel").glob("*.nc"))
    assert len(sequential_files) == len(FIXTURE_MODELS)
    assert sequential_files == parallel_files
    for name in FIXTURE_MODELS:
        assert sequential[name].fingerprint == parallel[name].fingerprint
        assert sequential[name].elpd_loo() == parallel[name].elpd_loo()
    pi.clear_fit_cache()


def test_fit_models_cached_raises_the_failure_not_a_cancellation(tmp_path, monkeypatch):
    """With ``stop_on_failure`` the pool cancels the fits not yet started when
    one fails; the caller must see the failure that caused it, whichever
    model's name sorts first."""
    models_dir, responses = _stub_models(tmp_path, ["a", "b", "c"])
    _stub_loading(monkeypatch)
    monkeypatch.setattr(
        pi, "_sample_models_in_pool",
        _fake_pool(
            [],
            failures={
                "a": CancelledError("fit of 'a' cancelled after 'c' failed"),
                "c": RuntimeError("bad initial energy"),
            },
        ),
    )

    with pytest.raises(RuntimeError, match="bad initial energy"):
        pi.fit_models_cached(
            ["a", "b", "c"], models_dir, responses, cache_dir=tmp_path / "cache",
            fit_workers=2,
        )


# ---------------------------------------------------------------------------
# What the worker must do for PyMC to sample inside a spawned process
# ---------------------------------------------------------------------------


@pytest.fixture
def _restore_start_method():
    import multiprocessing

    before = multiprocessing.get_start_method(allow_none=True)
    yield
    multiprocessing.set_start_method(before, force=True)


def test_worker_forks_its_chains(tmp_path, monkeypatch, _restore_start_method):
    """A spawned child's default start method is spawn, under which PyMC
    pickles the step method for its chain processes -- and the model lives in
    a module ``load_pymc_model`` executed from a file, which a fresh
    interpreter cannot import. The worker therefore forks its chains, as the
    sequential path does from the main process."""
    import multiprocessing

    models_dir, responses = _stub_models(tmp_path, ["a"])
    multiprocessing.set_start_method("spawn", force=True)
    seen = {}

    def fake_fit(name, *a, **k):
        seen["start_method"] = multiprocessing.get_start_method()
        return pi.FittedModel(name=name, model=object(), idata=object(), fingerprint="fp")

    monkeypatch.setattr(pi, "fit_model", fake_fit)

    pi._fit_model_in_worker("a", models_dir, responses, tmp_path / "cache", {})

    assert seen["start_method"] == "fork"


class _Awkward(Exception):
    """An exception the parent cannot rebuild: PyMC's ``ParallelSamplingError``
    takes a second constructor argument that pickling does not carry."""

    def __init__(self, message, chain):
        super().__init__(message)
        self.chain = chain


def test_worker_reraises_any_failure_as_one_the_parent_can_unpickle(
    tmp_path, monkeypatch, capsys
):
    import pickle

    models_dir, responses = _stub_models(tmp_path, ["a"])

    def fake_fit(name, *a, **k):
        raise _Awkward("Chain 0 failed with: bad initial energy", chain=0)

    monkeypatch.setattr(pi, "fit_model", fake_fit)

    with pytest.raises(pi.FitWorkerFailure) as info:
        pi._fit_model_in_worker("a", models_dir, responses, tmp_path / "cache", {})

    assert str(info.value) == "_Awkward: Chain 0 failed with: bad initial energy"
    assert str(pickle.loads(pickle.dumps(info.value))) == str(info.value)
    err = capsys.readouterr().err
    assert "Traceback" in err and "_Awkward" in err  # the original, in the run log


def test_fit_models_to_cache_reports_a_worker_failure_by_its_original_type(
    tmp_path, monkeypatch
):
    models_dir, responses = _stub_models(tmp_path, ["good", "bad"])
    _stub_loading(monkeypatch)
    monkeypatch.setattr(
        pi, "_sample_models_in_pool",
        _fake_pool(
            [], failures={"bad": pi.FitWorkerFailure("ParallelSamplingError: Chain 0 failed")}
        ),
    )

    failures = pi.fit_models_to_cache(
        ["good", "bad"], models_dir, responses, cache_dir=tmp_path / "cache", fit_workers=2
    )

    assert failures == {"bad": "ParallelSamplingError: Chain 0 failed"}


def test_fit_models_to_cache_reports_a_model_that_fails_before_the_pool(tmp_path):
    """Resolving a model's sampler settings executes its file in the parent.
    A model that raises there is that model's failure, not the batch's."""
    models_dir, responses = _stub_models(tmp_path, ["fine_stub"])
    (models_dir / "boom.py").write_text("raise RuntimeError('boom at import')\n", encoding="utf-8")

    failures = pi.fit_models_to_cache(
        ["boom", "fine_stub"], models_dir, responses, cache_dir=tmp_path / "cache",
        fit_workers=2, chains=1, cores=1,
    )

    assert failures["boom"] == "RuntimeError: boom at import"
    assert failures["fine_stub"].startswith("TypeError: ")


def test_fit_models_cached_raises_a_model_that_fails_before_the_pool(tmp_path):
    models_dir, responses = _stub_models(tmp_path, ["fine_stub"])
    (models_dir / "boom.py").write_text("raise RuntimeError('boom at import')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom at import"):
        pi.fit_models_cached(
            ["fine_stub", "boom"], models_dir, responses, cache_dir=tmp_path / "cache",
            fit_workers=2, chains=1, cores=1,
        )


def test_real_pool_reports_a_failure_the_parent_could_not_unpickle(tmp_path):
    """End to end through the spawn pool: a real model whose data-binding hook
    raises, inside the worker, an exception class defined in the model file
    with a two-argument constructor. The parent could rebuild neither the class
    nor the instance; the worker's wrapper is what makes the failure a report
    rather than a broken pool."""
    models_dir, responses = _stub_models(tmp_path, ["fine_stub"])
    (models_dir / "awkward.py").write_text(
        "import numpy as np\n"
        "import pymc as pm\n"
        "\n"
        "\n"
        "class ChainFailed(Exception):\n"
        "    def __init__(self, message, chain):\n"
        "        super().__init__(message)\n"
        "        self.chain = chain\n"
        "\n"
        "\n"
        "def prepare_observed(rows):\n"
        "    raise ChainFailed('chain 0 diverged while binding', 0)\n"
        "\n"
        "\n"
        "with pm.Model() as model:\n"
        "    chose_left = pm.Data('chose_left', np.zeros(1, dtype='int64'))\n"
        "    p = pm.Beta('p', 1.0, 1.0)\n"
        "    pm.Bernoulli('response', p=p, observed=chose_left)\n",
        encoding="utf-8",
    )

    failures = pi.fit_models_to_cache(
        ["awkward", "fine_stub"], models_dir, responses, cache_dir=tmp_path / "cache",
        fit_workers=2, chains=1, cores=1,
    )

    assert failures["awkward"] == "ChainFailed: chain 0 diverged while binding"
    assert failures["fine_stub"].startswith("TypeError: ")  # the stub has no `model`


def test_pool_driver_accepts_a_worker_that_refit_at_the_escalated_target_accept(
    tmp_path, monkeypatch
):
    """A fit that fails the convergence gate is refit at 0.95 inside the
    worker (``fit_model``), which then returns the refit's fingerprint. Both
    fits are on disk, and the parent's ``fit_model`` loads them the same way,
    so the pool must accept it rather than call it a harness fault."""
    models_dir, responses = _stub_models(tmp_path, ["a", "b"])
    _thread_executor(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    loop = {"target_accept": 0.8}
    escalated = {"target_accept": pi.ESCALATED_TARGET_ACCEPT}

    def refitting_fit(name, models_dir_, responses_, *, cache_dir, **kw):
        for kwargs in (loop, escalated):
            nc = Path(cache_dir) / _nc_name(name, models_dir, responses, kwargs)
            nc.touch()
        return pi.FittedModel(
            name=name, model=object(), idata=object(), fingerprint=nc.name.split(".")[1]
        )

    monkeypatch.setattr(pi, "fit_model", refitting_fit)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    outcomes = pi._sample_models_in_pool(
        ["a", "b"], models_dir, responses, cache_dir, loop,
        workers=2, stop_on_failure=True,
    )

    assert outcomes == {"a": None, "b": None}


# ---------------------------------------------------------------------------
# Infrastructure failures are not model failures
# ---------------------------------------------------------------------------
#
# One pool worker killed (an out-of-memory kill) raises BrokenProcessPool for
# every pending fit, and the start-of-experiment screen used to drop every one
# of those models — protected seeds included — as "MCMC fit failed".


def test_a_broken_pool_raises_instead_of_failing_every_pending_model(tmp_path, monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    models_dir, responses = _stub_models(tmp_path, ["a", "b", "c"])
    _thread_executor(monkeypatch)

    def dies(name, *args):
        raise BrokenProcessPool("A process in the process pool was terminated abruptly")

    monkeypatch.setattr(pi, "_fit_model_in_worker", dies)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with pytest.raises(pi.FitInfrastructureFailure, match="not the model's"):
        pi._sample_models_in_pool(
            ["a", "b", "c"], models_dir, responses, cache_dir, {},
            workers=2, stop_on_failure=False,
        )


def test_an_infrastructure_error_inside_a_worker_raises(tmp_path, monkeypatch):
    models_dir, responses = _stub_models(tmp_path, ["a", "b"])
    _thread_executor(monkeypatch)

    def fake_fit(name, *a, **k):
        raise MemoryError("Unable to allocate 3.1 GiB")

    monkeypatch.setattr(pi, "fit_model", fake_fit)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with pytest.raises(pi.FitInfrastructureFailure, match="MemoryError"):
        pi._sample_models_in_pool(
            ["a", "b"], models_dir, responses, cache_dir, {},
            workers=2, stop_on_failure=False,
        )


def test_the_tolerant_batch_raises_an_in_process_infrastructure_error(tmp_path, monkeypatch):
    models_dir, responses = _stub_models(tmp_path, ["good", "bad"])
    monkeypatch.setattr(pi, "_sample_models_in_pool", _never_pool)

    def fake_fit(name, *a, **k):
        if name == "bad":
            raise OSError(116, "Stale file handle")
        return pi.FittedModel(name=name, model=object(), idata=object(), fingerprint=name)

    monkeypatch.setattr(pi, "fit_model", fake_fit)

    with pytest.raises(OSError, match="Stale file handle"):
        pi.fit_models_to_cache(["good", "bad"], models_dir, responses, fit_workers=1)


def test_an_unreadable_cached_fit_is_an_infrastructure_failure(tmp_path, monkeypatch):
    """A truncated .nc (a write cut short) used to be read back as the model's
    failure on --resume."""
    models_dir, responses = _stub_models(tmp_path, ["a"])
    monkeypatch.setattr(pi, "load_pymc_model", lambda name, directory: object())
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / _nc_name("a", models_dir, responses, {})).write_bytes(b"\x89HDF\r\n truncated")

    with pytest.raises(pi.FitInfrastructureFailure, match="cannot be read"):
        pi.fit_model("a", models_dir, responses, cache_dir=cache_dir)


class _Idata:
    def __init__(self, fail: bool):
        self.fail = fail

    def to_netcdf(self, path):
        Path(path).write_bytes(b"half a fit")
        if self.fail:
            raise MemoryError("killed mid-write")


def test_a_fit_file_appears_only_once_it_is_complete(tmp_path):
    nc = tmp_path / "a.0123abcd.nc"
    with pytest.raises(MemoryError):
        pi.write_fit_file(_Idata(fail=True), nc)
    assert list(tmp_path.iterdir()) == []  # neither the fit nor a partial file

    pi.write_fit_file(_Idata(fail=False), nc)
    assert [p.name for p in tmp_path.iterdir()] == [nc.name]
    assert nc.read_bytes() == b"half a fit"



def test_predictions_that_are_not_probabilities_raise_with_the_affected_stimuli():
    import numpy as np

    raw = np.array([[[0.2, np.nan, 0.5], [0.3, 0.4, 1.5]]])  # (chain, draw, stimulus)
    with pytest.raises(pi.InvalidPredictions) as info:
        pi._validated_p_left_draws(raw, context="Model 'm' prior-predictive p_left")
    assert info.value.invalid_stimuli().tolist() == [False, True, True]
    assert info.value.draws.shape == (2, 3)
