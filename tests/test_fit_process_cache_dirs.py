"""Every process the fits start has a cache directory of its own.

On import, arviz 0.23 writes a once-a-day marker, ``<XDG cache>/arviz/
daily_warning``, through a fixed-name ``daily_warning.tmp`` whenever the
stored date is not today. Fit processes started together after midnight
collided on that file (``FileNotFoundError: daily_warning.tmp ->
daily_warning``) and ended the cell: 11 of 24 cells of the 2026-09-28
literature sweep. The race is made deterministic here: the shared marker is
stale and its temporary name is a directory, so any process that writes the
shared marker fails.
"""

from __future__ import annotations

import tempfile

import pytest

from src.models import pymc_inference as pi
from tests import fit_process_stand_ins as stand_ins
from tests.test_fit_time_limit import _request

STALE_DATE = "2000-01-01"


@pytest.fixture
def shared_cache(tmp_path, monkeypatch):
    """A shared XDG cache whose arviz marker every writer would fail on."""
    cache = tmp_path / "shared_cache"
    (cache / "arviz" / "daily_warning.tmp").mkdir(parents=True)
    (cache / "arviz" / "daily_warning").write_text(STALE_DATE)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    pi.clear_fit_cache()
    yield cache
    pi.clear_fit_cache()


def test_time_limited_fit_processes_import_arviz_in_caches_of_their_own(
    tmp_path, shared_cache
):
    requests = [_request(tmp_path, name) for name in ("a", "b", "c")]
    outcomes = pi.sample_fits_time_limited(
        requests,
        time_limit_sec=120,
        workers=3,
        _target=stand_ins.import_arviz_and_write_the_fit,
    )
    assert outcomes == [None, None, None]
    assert (shared_cache / "arviz" / "daily_warning").read_text() == STALE_DATE
    assert list((tmp_path / "tmp").iterdir()) == []  # the private caches are removed


def test_fit_pool_workers_import_arviz_in_caches_of_their_own(tmp_path, shared_cache):
    with (
        pi._fit_process_caches() as cache_root,
        pi._compile_dirs(2) as compile_dirs,
        pi._fit_executor(2, cache_root, compile_dirs) as pool,
    ):
        futures = [
            pool.submit(stand_ins.import_arviz_and_name_the_cache_dir) for _ in range(4)
        ]
        cache_dirs = {future.result() for future in futures}
    assert str(shared_cache) not in cache_dirs
    assert (shared_cache / "arviz" / "daily_warning").read_text() == STALE_DATE
    assert list((tmp_path / "tmp").iterdir()) == []
