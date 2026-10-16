"""Concurrent fit processes never wait on each other's PyTensor compile lock.

PyTensor compiles under one file lock per compile directory and gives up after
120 s. On 2026-09-28 every process of a node compiled in the one directory
``$L_SCRATCH/pytensor`` (``$L_SCRATCH`` is per user and node, not per job): a
cell's time-limited candidate fits, run four at a time, and those of the other
cells on the node. One cell died on ``FitInfrastructureFailure: ... Timeout:
The file lock '/lscratch/benpry/pytensor/compiledir_.../.lock' could not be
acquired``.

The contention is made deterministic here: the test holds the lock of the
directory every process would share, and each fit process takes the compile
lock with 2 s of patience. A process compiling in the shared directory fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from filelock import FileLock

from src.models import pymc_inference as pi
from src.runtime.config import REPO_ROOT
from tests import fit_process_stand_ins as stand_ins
from tests.test_fit_time_limit import _request

SLURM = (
    Path(__file__).resolve().parents[1] / "scripts" / "subjective_randomness" / "slurm"
)
LIVE = Path(__file__).resolve().parents[1] / "scripts" / "outer_loop_live"


@pytest.fixture
def shared_compiledir_locked(tmp_path, monkeypatch):
    """Every process's PyTensor base_compiledir is ``tmp_path/shared``, whose
    compile lock this test holds; fit slots live under ``tmp_path/slots``."""
    from pytensor import config

    shared_base = tmp_path / "shared"
    monkeypatch.setenv("PYTENSOR_FLAGS", f"base_compiledir={shared_base}")
    shared = shared_base / Path(config.compiledir).name
    shared.mkdir(parents=True)
    (tmp_path / "slots").mkdir()
    monkeypatch.setattr(pi, "_COMPILE_SLOTS_ROOT", tmp_path / "slots")
    monkeypatch.setattr(pi, "_COMPILE_SLOTS_IN_USE", set())
    pi.clear_fit_cache()
    with FileLock(shared / ".lock"):
        yield shared
    pi.clear_fit_cache()


def test_concurrent_time_limited_fits_compile_in_directories_of_their_own(
    tmp_path, shared_compiledir_locked
):
    requests = [_request(tmp_path, name) for name in ("a", "b", "c")]
    outcomes = pi.sample_fits_time_limited(
        requests,
        time_limit_sec=120,
        workers=3,
        _target=stand_ins.take_the_compile_lock_and_write_the_fit,
    )
    assert outcomes == [None, None, None]
    compiledirs = {(r.cache_dir / f"{r.name}.compiledir").read_text() for r in requests}
    assert len(compiledirs) == 3
    for compiledir in compiledirs:
        assert Path(compiledir).is_relative_to(tmp_path / "slots")


def test_a_fit_process_reuses_the_compile_directory_of_the_one_before_it(
    tmp_path, shared_compiledir_locked
):
    requests = [_request(tmp_path, name) for name in ("a", "b")]
    pi.sample_fits_time_limited(
        requests,
        time_limit_sec=120,
        workers=1,
        _target=stand_ins.take_the_compile_lock_and_write_the_fit,
    )
    first, second = (
        (r.cache_dir / f"{r.name}.compiledir").read_text() for r in requests
    )
    assert first == second


def test_fit_pool_workers_compile_in_directories_of_their_own(
    tmp_path, shared_compiledir_locked
):
    with (
        pi._fit_process_caches() as cache_root,
        pi._compile_dirs(2) as compile_dirs,
        pi._fit_executor(2, cache_root, compile_dirs) as pool,
    ):
        futures = [
            pool.submit(stand_ins.take_the_compile_lock_and_name_the_compile_dir)
            for _ in range(4)
        ]
        compiledirs = {future.result() for future in futures}
    assert 1 <= len(compiledirs) <= 2
    for compiledir in compiledirs:
        assert Path(compiledir).is_relative_to(tmp_path / "slots")


def test_compile_slots_are_not_shared_while_held(tmp_path, monkeypatch):
    monkeypatch.setattr(pi, "_COMPILE_SLOTS_ROOT", tmp_path)
    monkeypatch.setattr(pi, "_COMPILE_SLOTS_IN_USE", set())
    with pi._compile_dirs(2) as outer:
        with pi._compile_dirs(3) as inner:
            assert not set(outer) & set(inner)
        with pi._compile_dirs(1) as again:
            assert again == [inner[0]]  # released, and reused
    assert pi._COMPILE_SLOTS_IN_USE == set()


def test_the_compile_dir_flag_keeps_the_other_flags():
    assert (
        pi._flags_with_base_compiledir(
            "floatX=float64, base_compiledir=/shared,optimizer=fast_run", "/own"
        )
        == "floatX=float64,optimizer=fast_run,base_compiledir=/own"
    )
    assert pi._flags_with_base_compiledir("", "/own") == "base_compiledir=/own"
    with pytest.raises(RuntimeError, match="compiledir"):
        pi._flags_with_base_compiledir("compiledir=/pinned", "/own")


@pytest.mark.parametrize(
    "script",
    [
        SLURM / "holdout_recovery_array.sbatch",
        SLURM / "impossible_holdout_recovery_array.sbatch",
        LIVE / "run_live.sbatch",
    ],
)
def test_each_job_compiles_in_a_directory_of_its_own(script):
    """The harness process itself compiles too; jobs on one node must not share."""
    text = script.read_text(encoding="utf-8")
    compiledir_lines = [
        line
        for line in text.splitlines()
        if re.search(r"pytensor", line) and "SLURM_JOB_ID" in line
    ]
    assert compiledir_lines, f"{script.name}: the PyTensor compile dir is not per job"


def test_a_compile_directory_reached_through_a_symlink_is_its_own(tmp_path):
    """A home directory reached through a symlink (/sailhome/<u> and
    /afs/.../<u> on the Stanford CS machines) names the same compile
    directory two ways; the check compares where the paths lead."""
    import subprocess
    import sys

    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    script = (
        "from src.models.pymc_inference import _use_own_compile_dir\n"
        f"_use_own_compile_dir({str(tmp_path / 'link' / 'slot')!r})\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT)
    assert done.returncode == 0, done.stderr


def test_persistent_slots_are_claimed_across_processes_and_kept(tmp_path, monkeypatch):
    """With AUTO_PSYCH_COMPILE_SLOTS set (the coco cells), slots live under that
    root across jobs, so a new job starts on warm compiled code (a cold slot
    cost a fit ~80 s), and one held by another process (a file lock) is
    skipped, so no two processes ever share a compile lock."""
    import subprocess
    import sys

    root = tmp_path / "slots"
    root.mkdir()
    monkeypatch.setenv(pi.COMPILE_SLOTS_ENV, str(root))
    monkeypatch.setattr(pi, "_COMPILE_SLOTS_ROOT", None)
    holder = subprocess.Popen(
        [sys.executable, "-c", (
            "import fcntl, sys, time; f = open(sys.argv[1], 'w'); "
            "fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True); time.sleep(60)"
        ), str(root / "slot-0.lock")],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pi._compile_dirs(1) as [directory]:
            assert Path(directory) == root / "slot-1"
    finally:
        holder.kill()
        holder.wait()
    with pi._compile_dirs(1) as [directory]:
        assert Path(directory) == root / "slot-0"
    assert (root / "slot-0").is_dir() and (root / "slot-1").is_dir()  # kept for the next job
