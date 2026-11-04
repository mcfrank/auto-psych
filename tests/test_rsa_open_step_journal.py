"""Step mode's round journal and guards (src/rsa_open/loop/orchestrator.py), without fits."""

import json
import threading

import numpy as np
import pytest


def _bare(tmp_path):
    from src.rsa_open.loop.orchestrator import OpenLoop

    loop = OpenLoop.__new__(OpenLoop)
    loop._jpath, loop._jmemo, loop._jlock = None, {}, threading.Lock()
    return loop


def test_a_journaled_call_runs_once_and_a_continuation_reads_it_back(tmp_path, monkeypatch):
    import src.rsa_open.loop.slurm as step

    monkeypatch.setenv("RSA_OPEN_FIT_BACKEND", "slurm")
    monkeypatch.setattr(step, "_ACTIVE", object())
    calls = []

    def spawn():
        calls.append(1)
        return {"ok": np.bool_(True), "elpd": np.float64(-1.5), "worst": -np.inf}

    first = _bare(tmp_path)
    first._open_journal(tmp_path / "round_1")
    assert first._memo("admit:x", spawn) == spawn() and len(calls) == 2  # (the second call is the comparison's)
    assert first._memo("admit:x", spawn)["elpd"] == -1.5 and len(calls) == 2

    later = _bare(tmp_path)  # a continuation: a new process, the same round
    later._open_journal(tmp_path / "round_1")
    out = later._memo("admit:x", spawn)
    assert len(calls) == 2 and out == {"ok": True, "elpd": -1.5, "worst": -np.inf}
    assert json.loads((tmp_path / "round_1" / "journal.json").read_text())["admit:x"]["ok"] is True


def test_a_failed_call_is_not_journaled(tmp_path, monkeypatch):
    import src.rsa_open.loop.slurm as step

    monkeypatch.setenv("RSA_OPEN_FIT_BACKEND", "slurm")
    monkeypatch.setattr(step, "_ACTIVE", object())
    loop = _bare(tmp_path)
    loop._open_journal(tmp_path / "round_1")

    def pending():
        raise RuntimeError("fits pending")

    with pytest.raises(RuntimeError):
        loop._memo("start", pending)
    assert loop._memo("start", lambda: 3) == 3


def test_a_local_run_keeps_no_journal(tmp_path, monkeypatch):
    monkeypatch.delenv("RSA_OPEN_FIT_BACKEND", raising=False)
    loop = _bare(tmp_path)
    loop._open_journal(tmp_path / "round_1")
    calls = []
    loop._memo("k", lambda: calls.append(1))
    loop._memo("k", lambda: calls.append(1))
    assert len(calls) == 2 and not (tmp_path / "round_1" / "journal.json").exists()


def test_step_mode_refuses_a_local_fit_backend(tmp_path, monkeypatch):
    import src.rsa_open.loop.slurm as step

    monkeypatch.delenv("RSA_OPEN_FIT_BACKEND", raising=False)
    monkeypatch.setattr(step, "_ACTIVE", object())
    with pytest.raises(RuntimeError, match="RSA_OPEN_FIT_BACKEND=slurm"):
        _bare(tmp_path)._step_mode()


def test_a_yield_waits_for_every_array_the_step_submitted_early(tmp_path):
    from src.rsa_open.loop.fit_array import FitsPending
    from src.rsa_open.loop.slurm import Stepper, StepYield
    from tests.test_rsa_open_step_submit import FakeSlurm, _cfg, _pending

    slurm = FakeSlurm()
    stepper = Stepper(_cfg(tmp_path), slurm)
    folds = _pending(tmp_path, sizes=(18,))
    stepper.submit_early(folds, "folds")  # array 101
    repairs = FitsPending([p for p in _pending(tmp_path / "r", sizes=(18,)).pending])
    with pytest.raises(StepYield) as y:
        stepper.yield_step(repairs, "repairs")  # array 102, not the folds
    assert y.value.record["after_early"] == ["101"]
    assert slurm.calls[-2][-1] == "afterany:102:101,singleton"
