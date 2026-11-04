"""Submitting a step's fits and continuation (src/rsa_open/loop/slurm.py), against a stand-in sbatch."""

import json

import pytest


class FakeSlurm:
    def __init__(self, queued=0, fail_on=None):
        self.calls, self.queued, self.fail_on, self.next = [], queued, fail_on, 100

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        if self.fail_on and self.fail_on in " ".join(cmd):
            raise RuntimeError(f"{cmd[0]} failed")
        if cmd[0] == "squeue":
            return "".join(f"{i}\n" for i in range(self.queued))
        if cmd[0] in ("sbatch", "bash"):
            self.next += 1
            return f"{self.next}\n"
        return ""


def _cfg(tmp_path, **kw):
    from src.rsa_open.loop.slurm import StepConfig

    cell = tmp_path / "cells" / "open_chain_1"
    cell.mkdir(parents=True)
    return StepConfig(cell, "rsa_open_open_c1", cell / "continuation.sh", tmp_path / "fit_array.sbatch",
                      {"VENV_PY": "/v/bin/python", "HARNESS_REPO": "/h", "JAX_COMPILATION_CACHE_DIR": "/j"},
                      tmp_path / "logs", **kw)


def _pending(tmp_path, sizes=(18, 18, 36), limit=3600.0):
    from src.rsa.fit import FitSettings
    from src.rsa_open.loop.fit_array import FitPending, FitsPending
    from src.rsa_open.loop.fitting import FitJob

    return FitsPending([FitPending(FitJob(tmp_path / f"t{i}.py", f"t{i}", tmp_path / "c.json", FitSettings(),
                                          tmp_path / "cache" / f"t{i}.nc", tmp_path / "cache" / f"t{i}.json", limit), gb)
                        for i, gb in enumerate(sizes)])


def test_held_arrays_by_memory_then_the_continuation_after_them_then_release(tmp_path):
    from src.rsa_open.loop.slurm import submit_step

    cfg, slurm = _cfg(tmp_path), FakeSlurm()
    record = submit_step(cfg, _pending(tmp_path), runner=slurm, reason="round 2: candidates")
    kinds = [c[0] for c in slurm.calls]
    assert kinds == ["squeue", "sbatch", "sbatch", "bash", "scontrol"]
    small, large = slurm.calls[1], slurm.calls[2]
    assert "--hold" in small and "--array=0-1%24" in small and "--mem=18G" in small and "--mem=36G" in large
    assert "--cpus-per-task=1" in small and "--partition=mcfrank,owners" in small and "--constraint=CPU_GEN:SIE" in small
    assert "--time=80" in small  # the fits' 60-minute limit and a 20-minute margin
    export = next(a for a in small if a.startswith("--export="))
    assert export.startswith("--export=NONE,") and "VENV_PY=/v/bin/python" in export and "SECRET" not in export.upper()
    assert slurm.calls[3] == ["bash", str(cfg.continuation), "afterany:101:102,singleton"]
    assert slurm.calls[4] == ["scontrol", "release", "101,102"]
    assert record["continuation"] == "103" and record["tasks"] == {"18": 2, "36": 1}
    manifest = json.loads(next(cfg.arrays_dir.glob("*.mem18.json")).read_text())
    assert manifest["mem_gb"] == 18 and [t["name"] for t in manifest["tasks"]] == ["t0", "t1"]
    assert len(cfg.steps_log.read_text().splitlines()) == 1


def test_a_failed_submission_cancels_what_it_submitted_and_stops_the_chain(tmp_path):
    from src.rsa_open.loop.slurm import submit_step

    cfg, slurm = _cfg(tmp_path), FakeSlurm(fail_on="continuation.sh")
    with pytest.raises(RuntimeError, match="submitting the step failed"):
        submit_step(cfg, _pending(tmp_path), runner=slurm)
    assert [c for c in slurm.calls if c[0] == "scancel"] == [["scancel", "101"], ["scancel", "102"]]
    assert "submission failed" in (cfg.cell_dir / "STOP").read_text()
    assert not cfg.steps_log.exists()


def test_a_step_refuses_when_stopped_over_budget_or_the_queue_is_full(tmp_path):
    from src.rsa_open.loop.slurm import StepRefused, submit_step

    cfg = _cfg(tmp_path, max_steps=1, max_queued=10)
    submit_step(cfg, _pending(tmp_path, sizes=(18,)), runner=FakeSlurm())
    with pytest.raises(StepRefused, match="step budget spent"):
        submit_step(cfg, _pending(tmp_path, sizes=(18,)), runner=FakeSlurm())
    assert "step budget" in (cfg.cell_dir / "STOP").read_text()
    with pytest.raises(StepRefused, match="is stopped"):
        submit_step(cfg, _pending(tmp_path, sizes=(18,)), runner=FakeSlurm())
    (cfg.cell_dir / "STOP").unlink()
    cfg2 = _cfg(tmp_path / "b", max_queued=10)
    slurm = FakeSlurm(queued=9)
    with pytest.raises(StepRefused, match="already queued"):
        submit_step(cfg2, _pending(tmp_path, sizes=(18, 18)), runner=slurm)
    assert [c[0] for c in slurm.calls] == ["squeue"]  # nothing submitted


def test_a_fit_without_a_time_limit_gets_the_partition_s_day_limit(tmp_path):
    from src.rsa_open.loop.slurm import submit_step

    slurm = FakeSlurm()
    submit_step(_cfg(tmp_path), _pending(tmp_path, sizes=(18,), limit=float("inf")), runner=slurm)
    assert "--time=2-00:00:00" in slurm.calls[1]


def test_the_fit_task_script_never_reads_the_secrets():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "scripts" / "rsa" / "slurm" / "fit_array.sbatch").read_text()
    assert "_env.sh" not in text.split("set -euo pipefail")[1] and ".secrets" not in text.split("set -euo pipefail")[1]
    assert "src.rsa_open.loop.fit_array" in text
