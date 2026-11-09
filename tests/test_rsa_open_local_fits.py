"""Fits in the step's own job (PI 2026-11-09; src/rsa_open/loop/slurm.py `LocalStepper`). The stepped
rehearsal (review V3 §16) found 316 of 325 fit tasks under rule 10's 10-minute line, so a step runs
its fits itself: the arrays' task lists and result files, several tasks at once on the job's cores,
folds after every fit started before them, then the chain continues in the same job. A launch may
run every experiment (LAUNCH_EXPERIMENTS=all)."""

import json
import threading
import time
from pathlib import Path

import pytest

from src.rsa.fit import FitSettings


def _cfg(cell: Path):
    from src.rsa_open.loop.slurm import StepConfig

    cell.mkdir(parents=True, exist_ok=True)
    return StepConfig(cell, "rsa_open_test", cell / "continuation.sh", cell / "fit_array.sbatch",
                      {"VENV_PY": "/v", "HARNESS_REPO": "/h", "JAX_COMPILATION_CACHE_DIR": "/j"}, cell / "logs")


def _pending(root: Path, name: str, gb: int = 18):
    from src.rsa_open.loop.fit_array import FitPending
    from src.rsa_open.loop.fitting import FitJob

    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    job = FitJob(root / f"{name}.py", name, root / "corpus.json", FitSettings(), cache / f"{name}.nc",
                 cache / f"{name}.json", 600.0)
    return FitPending(job, gb)


def _after(root: Path, name: str, fold: int):
    from src.rsa_open.loop.fit_array import FoldAfter

    return FoldAfter(root / f"{name}.py", name, root / "corpus.json", FitSettings(), root / "cache",
                     root / f"fold{fold}.json", FitSettings(), {"fixed": 600.0}, False, 18)


class Launcher:
    """Records when each task ran (manifest kind, index); sleeps so overlap is visible."""

    def __init__(self, sleep=0.05):
        self.runs, self.lock, self.sleep = [], threading.Lock(), sleep

    def __call__(self, manifest: Path, index: int) -> int:
        start = time.monotonic()
        time.sleep(self.sleep)
        with self.lock:
            self.runs.append((manifest.name.split("_")[0], index, start, time.monotonic()))
        return 0


def test_workers_take_a_core_each_and_leave_the_step_its_core_and_memory():
    from src.rsa_open.loop.slurm import local_workers

    assert local_workers(cores=12, memory_gb=72) == 8  # (72 - 24) / 6, under 11 cores
    assert local_workers(cores=4, memory_gb=200) == 3  # the step keeps a core
    assert local_workers(cores=1, memory_gb=16) == 1  # always at least one


def test_a_step_runs_its_fits_folds_after_them_and_continues(tmp_path):
    from src.rsa_open.loop.fit_array import FitsPending
    from src.rsa_open.loop.slurm import LocalStepper, StepContinue

    launcher = Launcher()
    stepper = LocalStepper(_cfg(tmp_path / "cell"), workers=3, launcher=launcher)
    early = FitsPending([_pending(tmp_path, "a"), _pending(tmp_path, "b")], [_after(tmp_path, "a", 0)])
    stepper.submit_early(early, "repairs at work")
    later = FitsPending([_pending(tmp_path, "a"), _pending(tmp_path, "c", 36)], [_after(tmp_path, "c", 1)])
    with pytest.raises(StepContinue) as cont:
        stepper.yield_step(later, "test")
    record = cont.value.record
    assert record["local"] and record["tasks"] == 5 and record["failed_tasks"] == 0  # a, b, c once each; 2 folds
    fits = [r for r in launcher.runs if r[0] == "fits"]
    folds = [r for r in launcher.runs if r[0] == "after"]
    assert len(fits) == 3 and len(folds) == 2
    first_fold = min(r[2] for r in folds)
    early_fits_end = sorted(r[3] for r in fits)[1]  # a and b, started before the first fold
    assert first_fold >= early_fits_end  # a fold waits for every fit started before it
    log = [json.loads(line) for line in (tmp_path / "cell" / "steps.jsonl").read_text().splitlines()]
    assert [e.get("early", False) for e in log] == [True, False] and all(e["local"] for e in log)
    manifests = sorted(p.name.split("_")[0] for p in (tmp_path / "cell").rglob("*.json") if p.parent.name != "cell")
    assert manifests.count("fits") == 2 and manifests.count("after") == 2  # one per batch and memory size, as arrays
    assert (tmp_path / "cache" / "a.nc.mem18.submitted").read_text() == "1"  # counted once, as an array would
    stepper.close()


def test_a_launch_s_step_budget_counts_steps_run_in_the_job(tmp_path):
    from src.rsa_open.loop.fit_array import FitsPending
    from src.rsa_open.loop.slurm import LocalStepper, StepContinue, StepRefused, begin_launch

    cfg = _cfg(tmp_path / "cell")
    begin_launch(cfg, 1, max_steps=1)
    stepper = LocalStepper(cfg, workers=1, launcher=Launcher(0.0))
    with pytest.raises(StepContinue):
        stepper.yield_step(FitsPending([_pending(tmp_path, "a")]), "first")
    with pytest.raises(StepRefused, match="budget"):
        stepper.yield_step(FitsPending([_pending(tmp_path, "b")]), "second")
    assert (tmp_path / "cell" / "STOP").exists()
    stepper.close()


def test_a_launch_may_run_every_experiment(tmp_path):
    from src.rsa_open.loop.slurm import MAX_STEPS, begin_launch

    one = begin_launch(_cfg(tmp_path / "one"), 2)
    assert one["last_experiment"] == 2 and one["max_steps"] == MAX_STEPS
    every = begin_launch(_cfg(tmp_path / "all"), 1, last_experiment=3)
    assert every["last_experiment"] == 3 and every["max_steps"] == 3 * MAX_STEPS


def test_step_mode_runs_fits_in_the_job_unless_told_to_submit_them(tmp_path, monkeypatch):
    import src.rsa_open.loop.slurm as step
    from src.rsa_open.run_chain import step_from_env

    for k, v in {"RSA_OPEN_STEP_MODE": "1", "RSA_OPEN_FIT_BACKEND": "slurm", "RSA_OPEN_CELL_DIR": str(tmp_path),
                 "RSA_OPEN_CHAIN_JOB": "j", "RSA_OPEN_CONTINUATION": str(tmp_path / "c.sh"),
                 "RSA_OPEN_FIT_SCRIPT": "f", "RSA_OPEN_LOGS": str(tmp_path), "VENV_PY": "/v", "HARNESS_REPO": "/h",
                 "JAX_COMPILATION_CACHE_DIR": "/j"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv(step.STEP_FITS_ENV, raising=False)
    try:
        assert type(step_from_env()) is step.LocalStepper
        monkeypatch.setenv(step.STEP_FITS_ENV, "array")
        assert type(step_from_env()) is step.Stepper
        monkeypatch.setenv(step.STEP_FITS_ENV, "elsewhere")
        with pytest.raises(RuntimeError, match="local .* or array"):
            step_from_env()
    finally:
        step.activate(None)


def test_a_local_fit_gets_a_clean_environment_and_a_core_of_its_own(tmp_path, monkeypatch):
    import subprocess

    import src.rsa_open.loop.slurm as step

    seen = {}

    def fake_run(cmd, env, cwd, capture_output, text):
        import os

        seen.update(cmd=cmd, env=env, cwd=cwd, cores=os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(step.subprocess, "run", fake_run)
    for k, v in {"VENV_PY": "/venv/python", "HARNESS_REPO": str(tmp_path), "JAX_COMPILATION_CACHE_DIR": "/jc",
                 "GOOGLE_API_KEY": "secret", "RSA_OPEN_STEP_MODE": "1"}.items():
        monkeypatch.setenv(k, v)
    assert step.run_local_task(tmp_path / "m.json", 3) == 0
    assert seen["cmd"][:3] == ["/venv/python", "-m", "src.rsa_open.loop.fit_array"] and seen["cmd"][-1] == "3"
    assert "GOOGLE_API_KEY" not in seen["env"] and not any(k.startswith("RSA_OPEN_") for k in seen["env"])
    assert seen["env"]["JAX_COMPILATION_CACHE_DIR"] == "/jc" and seen["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert seen["cwd"] == str(tmp_path)
    if seen["cores"] is not None:
        assert len(seen["cores"]) == 1  # started from a thread pinned to one core of the job
