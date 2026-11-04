"""Fits as Slurm array tasks (src/rsa_open/loop/fit_array.py): a pending fit, its task, its result."""

import json

import pytest

from tests.test_rsa_open_loop_fitting import QUICK, corpus, theory_path  # noqa: F401 (fixtures)


def _job(tmp_path, name="t", limit=600.0):
    from src.rsa.fit import FitSettings
    from src.rsa_open.loop.fitting import FitJob

    return FitJob(tmp_path / "t.py", name, tmp_path / "c.json", FitSettings(), tmp_path / "cache" / f"{name}.fp.nc",
                  tmp_path / "cache" / f"{name}.fp.json", limit)


def _result(job, gb, record):
    from src.rsa_open.loop.fit_array import result_path

    path = result_path(job, gb)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def test_a_fit_without_a_result_is_pending_and_each_result_reads_back_as_the_local_backend_s(tmp_path):
    from src.rsa.loop.fitting import FitTimeLimitExceeded
    from src.rsa_open.loop.fit_array import FIT_GB, RETRY_GB, ArrayBackend, FitPending
    from src.rsa_open.loop.fitting import _Killed

    backend, job = ArrayBackend(), _job(tmp_path)
    with pytest.raises(FitPending) as p:
        backend.attempt(job)
    assert p.value.mem_gb == FIT_GB
    _result(job, FIT_GB, {"kind": "model", "message": "ValueError: the theory's own", "traceback": "tb"})
    assert backend.attempt(job) == ("model", "ValueError: the theory's own", "tb")
    _result(job, FIT_GB, {"kind": "timeout", "message": "t: the fit did not finish within 10 minutes"})
    with pytest.raises(FitTimeLimitExceeded, match="did not finish within 10 minutes"):
        backend.attempt(job)
    _result(job, FIT_GB, {"kind": "killed", "exitcode": -9, "peak": 17 * 2**30})
    with pytest.raises(_Killed) as k:
        backend.attempt(job)
    assert k.value.reserved == FIT_GB * 2**30 and k.value.peak == 17 * 2**30
    with pytest.raises(FitPending) as p:  # its retry, with twice the room, is the larger size
        backend.attempt(job, at_least=2 * FIT_GB * 2**30)
    assert p.value.mem_gb == RETRY_GB
    _result(job, FIT_GB, {"kind": "ok"})
    with pytest.raises(RuntimeError, match="isn't there"):  # an ok without its fit
        backend.attempt(job)


def test_a_fit_submitted_twice_without_a_result_is_an_infrastructure_failure(tmp_path):
    from src.rsa_open.loop.fit_array import ArrayBackend, FitPending, FitsPending, write_manifest

    backend, job = ArrayBackend(), _job(tmp_path)
    for k in (1, 2):
        with pytest.raises(FitPending) as p:
            backend.attempt(job)
        write_manifest(tmp_path / "arrays" / f"a{k}.json", [p.value])
    with pytest.raises(RuntimeError, match="submitted 2 times"):
        backend.attempt(job)
    other = _job(tmp_path, "u")
    with pytest.raises(FitPending) as q:
        backend.attempt(other, at_least=30 * 2**30)
    with pytest.raises(ValueError, match="one memory size per array"):
        write_manifest(tmp_path / "arrays" / "mixed.json", [p.value, q.value])
    assert len(FitsPending([p.value, p.value, q.value]).pending) == 2  # one task per fit and size


def test_a_phase_s_pending_fits_are_all_collected_before_the_step_yields(tmp_path):
    from src.rsa_open.loop.fit_array import ArrayBackend, FitsPending, collect_pending

    backend = ArrayBackend()
    calls = [lambda: backend.attempt(_job(tmp_path, "a")), lambda: "done", lambda: backend.attempt(_job(tmp_path, "b"))]
    with pytest.raises(FitsPending) as ps:
        collect_pending(calls)
    assert sorted(p.job.name for p in ps.value.pending) == ["a", "b"]
    assert collect_pending([lambda: 1, lambda: 2]) == [1, 2]


def test_an_array_task_fits_as_the_local_child_does_and_the_loop_reads_its_fit(theory_path, corpus, tmp_path,
                                                                              monkeypatch):
    from src.rsa.loop.fitting import cache_paths
    from src.rsa_open.loop.fit_array import FitsPending, collect_pending, run_task, write_manifest
    from src.rsa_open.loop.fitting import FIT_BACKEND_ENV, fingerprint, fit_cached

    _, path = corpus
    monkeypatch.setenv(FIT_BACKEND_ENV, "slurm")
    with pytest.raises(FitsPending) as ps:
        collect_pending([lambda: fit_cached(theory_path, "rsa_l1", path, QUICK, tmp_path / "cache", time_limit_sec=600)])
    manifest = write_manifest(tmp_path / "arrays" / "a.json", ps.value.pending)
    record = run_task(manifest, 0)  # what the array task does
    assert record["kind"] == "ok" and record["machine"]["host"]
    fitted = fit_cached(theory_path, "rsa_l1", path, QUICK, tmp_path / "cache", time_limit_sec=600)
    assert fitted.idata.posterior["alpha"].size == 40
    nc, meta = cache_paths(tmp_path / "cache", "rsa_l1", fingerprint(theory_path, path, QUICK))
    assert nc.exists() and json.loads(meta.read_text())["peak_hwm_mb"] > 0
    assert run_task(manifest, 0) == record  # a requeued task whose result is in does nothing
