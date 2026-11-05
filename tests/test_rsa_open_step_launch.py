"""Step mode as SRCC's ticket #106983 shapes it (2026-11-05), and the Sherlock rehearsal's findings
(review V3 §11): one experiment per launch with a budget of steps, folds that follow their full fits,
memory sized by a theory's recorded peak, no fit inside a step, and a killed session run again clean."""

import json
import threading
import types
from pathlib import Path

import pytest

from tests.test_rsa_open_fit_array import _job, _result
from tests.test_rsa_open_step_submit import FakeSlurm, _cfg, _pending


# ---------- memory ----------

def test_a_theory_with_a_recorded_peak_starts_at_the_tier_that_holds_it(tmp_path):
    from src.rsa_open.loop.fit_array import ArrayBackend, FitPending, mem_gb_for
    from src.rsa_open.loop.fitting import scheduler

    assert [mem_gb_for(g * 2**30) for g in (0, 18, 19, 36, 37, 72)] == [18, 18, 36, 36, 72, 72]
    with pytest.raises(RuntimeError, match="largest tier"):
        mem_gb_for(73 * 2**30)
    job = _job(tmp_path, name="peaky_theory")
    scheduler().observe("peaky_theory", 30 * 2**30)  # chain 1's experiment-3 folds: 28-32 GB
    with pytest.raises(FitPending) as p:
        ArrayBackend().attempt(job)
    assert p.value.mem_gb == 72  # 30 GB with margin is over 36


def test_a_result_at_a_lower_tier_stands_unless_the_fit_was_killed_there(tmp_path):
    from src.rsa_open.loop.fit_array import ArrayBackend, FitPending
    from src.rsa_open.loop.fitting import scheduler

    job = _job(tmp_path, name="grown_theory")
    _result(job, 18, {"kind": "model", "message": "the theory's own", "traceback": ""})
    scheduler().observe("grown_theory", 20 * 2**30)  # learnt after the fit was submitted at 18 GB
    assert ArrayBackend().attempt(job)[0] == "model"
    _result(job, 18, {"kind": "killed", "exitcode": -9, "peak": 17 * 2**30})
    with pytest.raises(FitPending) as p:
        ArrayBackend().attempt(job)
    assert p.value.mem_gb == 36


def test_an_unlimited_fit_in_step_mode_is_an_array_task_not_a_fit_in_the_step(tmp_path, monkeypatch):
    from src.rsa.fit import FitSettings
    from src.rsa_open.loop import fitting
    from src.rsa_open.loop.fit_array import FitPending

    monkeypatch.setenv("RSA_OPEN_FIT_BACKEND", "slurm")
    monkeypatch.setattr(fitting, "_fit_now", lambda *a, **k: pytest.fail("fitted inside the step"))
    (tmp_path / "t.py").write_text("x = 1\n")
    (tmp_path / "c.json").write_text("{}")
    with pytest.raises(FitPending) as p:
        fitting.fit_cached(tmp_path / "t.py", "t", tmp_path / "c.json", FitSettings(), tmp_path / "cache",
                           time_limit_sec=None)
    assert p.value.job.time_limit_sec == float("inf")
    from src.rsa_open.loop.slurm import _time_arg

    assert _time_arg([p.value.job.time_limit_sec]) == "2-00:00:00"


# ---------- folds after their full fits ----------

def _after(tmp_path, n=2, gb=18):
    from src.rsa.fit import FitSettings
    from src.rsa_open.loop.fit_array import FoldAfter

    return [FoldAfter(tmp_path / "t0.py", "t0", tmp_path / "c.json", FitSettings(), tmp_path / "cache",
                      tmp_path / f"fold_{f}.json", FitSettings(num_warmup=300, num_samples=500), {"fixed": 600.0},
                      False, gb) for f in range(n)]


def test_folds_after_their_fits_start_when_those_arrays_end_and_the_next_step_waits_for_both(tmp_path):
    from src.rsa_open.loop.fit_array import FitsPending
    from src.rsa_open.loop.slurm import Stepper, StepYield

    slurm = FakeSlurm()
    stepper = Stepper(_cfg(tmp_path), slurm)
    with pytest.raises(StepYield) as y:
        stepper.yield_step(FitsPending(_pending(tmp_path, sizes=(18,)).pending, _after(tmp_path)), "seeds")
    sb = [c for c in slurm.calls if c[0] == "sbatch"]
    assert len(sb) == 2 and "--hold" in sb[1] and "--dependency=afterany:101" in sb[1]
    assert json.loads(Path(sb[1][-1]).read_text())["tasks"][0]["kind"] == "fold_after"
    assert slurm.calls[-2] == ["bash", str(stepper.cfg.continuation), "afterany:101:102,singleton"]
    assert slurm.calls[-1] == ["scontrol", "release", "101,102"]
    assert y.value.record["folds_after"] == 2


def test_a_fold_after_task_fits_nothing_without_its_full_fit(tmp_path):
    from src.rsa_open.loop.fit_array import after_record, run_fold_after

    (tmp_path / "t0.py").write_text("x = 1\n")
    (tmp_path / "c.json").write_text("{}")
    out = run_fold_after(json.loads(json.dumps(after_record(_after(tmp_path, n=1)[0]))))
    assert out["kind"] == "skipped" and "not in the cache" in out["message"]


def test_collecting_keeps_the_folds_that_follow(tmp_path):
    from src.rsa_open.loop.fit_array import FitsPending, collect_pending

    def a():
        raise FitsPending(_pending(tmp_path, sizes=(18,)).pending, _after(tmp_path))

    def b():
        raise FitsPending(_pending(tmp_path / "b", sizes=(18,)).pending, _after(tmp_path))  # the same folds

    with pytest.raises(FitsPending) as ps:
        collect_pending([a, b, lambda: 3])
    assert len(ps.value.pending) == 2 and len(ps.value.after) == 2


# ---------- one experiment per launch ----------

def test_the_step_budget_counts_from_the_launch(tmp_path):
    from src.rsa_open.loop.slurm import StepRefused, begin_launch, current_launch, submit_step

    cfg = _cfg(tmp_path, max_steps=40)
    cfg.steps_log.write_text("".join(json.dumps({"continuation": str(i)}) + "\n" for i in range(45)))
    begin_launch(cfg, 2, max_steps=2)  # 45 steps before it, in earlier launches
    assert current_launch(cfg)["steps_before"] == 45
    slurm = FakeSlurm()
    for _ in range(2):
        submit_step(cfg, _pending(tmp_path, sizes=(18,)), runner=slurm)
    with pytest.raises(StepRefused, match="in experiment 2's launch"):
        submit_step(cfg, _pending(tmp_path, sizes=(18,)), runner=slurm)
    assert "step budget spent" in (cfg.cell_dir / "STOP").read_text()
    assert len((cfg.cell_dir / "launches.jsonl").read_text().splitlines()) == 1


def test_a_launch_runs_one_experiment_and_stops_for_a_person(tmp_path, monkeypatch):
    from src.rsa_open.chain import Chain, ChainConfig, ExperimentDone

    run = tmp_path / "run"
    (run / "experiment1" / "loop").mkdir(parents=True)
    (run / "experiment1" / "loop" / "export.json").write_text("{}")
    chain = Chain(ChainConfig(chain="chain_0", seeds_dir=tmp_path, literature=None, out_dir=run, ground_truth=None,
                              n_experiments=3), designer_spawn=None, theorist_spawn=None)
    assert chain.first_unfinished() == 2
    for n in (1, 2, 3):
        d = run / f"experiment{n}"
        (d / "design" / "frozen").mkdir(parents=True, exist_ok=True)
        (d / "design" / "frozen" / "record.json").write_text("{}")
        (d / "data").mkdir(exist_ok=True)
        (d / "data" / "block.json").write_text("{}")
    loops = []
    monkeypatch.setattr(Chain, "design", lambda self, n: None)
    monkeypatch.setattr(Chain, "collect", lambda self, n: None)

    def model_loop(self, n):
        loops.append(n)
        (self.exp(n) / "loop").mkdir(exist_ok=True)
        (self.exp(n) / "loop" / "export.json").write_text("{}")
        return {}

    monkeypatch.setattr(Chain, "model_loop", model_loop)
    from dataclasses import replace

    chain.cfg = replace(chain.cfg, stop_after_experiment=2)
    with pytest.raises(ExperimentDone, match="experiment 2 of 3 is done; launch experiment 3"):
        chain.run()
    assert loops == [1, 2]  # experiment 1's export is read back, not rerun (model_loop returns it)
    chain.cfg = replace(chain.cfg, stop_after_experiment=3)
    chain.run()  # the last experiment ends the chain, not the launch


def test_a_new_launch_records_its_experiment_and_a_continuation_reads_it(tmp_path, monkeypatch):
    from src.rsa_open.run_chain import this_launch

    cfg = _cfg(tmp_path)
    chain = types.SimpleNamespace(first_unfinished=lambda: 2, cfg=types.SimpleNamespace(n_experiments=4, chain="chain_0"))
    monkeypatch.setenv("RSA_OPEN_NEW_LAUNCH", "0")
    with pytest.raises(RuntimeError, match="no launch record"):
        this_launch(cfg, chain)
    monkeypatch.setenv("RSA_OPEN_NEW_LAUNCH", "1")
    assert this_launch(cfg, chain)["experiment"] == 2
    monkeypatch.setenv("RSA_OPEN_NEW_LAUNCH", "0")
    chain.first_unfinished = lambda: 3  # a continuation keeps its launch's experiment
    assert this_launch(cfg, chain)["experiment"] == 2


def test_a_finished_experiment_exits_8(tmp_path, monkeypatch):
    import src.rsa_open.run_chain as rc
    from src.rsa_open.chain import ExperimentDone
    from src.rsa_open.loop import slurm as step

    stepper = step.Stepper(_cfg(tmp_path), FakeSlurm())

    def run():
        raise ExperimentDone(1, 4)

    chain = types.SimpleNamespace(run=run, cfg=types.SimpleNamespace(), first_unfinished=lambda: 1)
    monkeypatch.setattr(rc, "build", lambda args: chain)
    monkeypatch.setattr(rc, "step_from_env", lambda: stepper)
    monkeypatch.setattr(rc, "this_launch", lambda cfg, ch: {"experiment": 1})
    monkeypatch.setattr(rc, "replace", lambda cfg, **kw: cfg)
    with pytest.raises(SystemExit) as out:
        rc.main(types.SimpleNamespace(run_dir=tmp_path / "run"))
    assert out.value.code == rc.EXIT_EXPERIMENT_DONE == 8


def test_before_its_first_scored_step_a_stepped_loop_is_cleared_not_set_aside(tmp_path):
    from src.rsa_open.chain import clear_stage

    loop = tmp_path / "loop"
    for name in (".fit_cache/a.nc", ".scores/b.json", "corpus.json", "models/x.py"):
        (loop / name).parent.mkdir(parents=True, exist_ok=True)
        (loop / name).write_text("x")
    clear_stage(loop, keep=(".fit_cache", ".scores"))
    assert sorted(p.name for p in loop.iterdir()) == [".fit_cache", ".scores"]
    assert not list(tmp_path.glob("loop.abandoned_*"))


# ---------- a killed session ----------

def test_a_session_killed_with_its_step_runs_again_from_its_brief_alone(tmp_path, monkeypatch):
    import src.rsa_open.loop.slurm as step
    from src.rsa_open.loop.orchestrator import OpenLoop

    monkeypatch.setenv("RSA_OPEN_FIT_BACKEND", "slurm")
    monkeypatch.setattr(step, "_ACTIVE", object())

    def bare(spawn):
        loop = OpenLoop.__new__(OpenLoop)
        loop._jpath, loop._jmemo, loop._jlock, loop.spawn = None, {}, threading.Lock(), spawn
        loop._open_journal(tmp_path / "round_1")
        return loop

    cdir = tmp_path / "round_1" / "candidate_1"
    cdir.mkdir(parents=True)
    (cdir / "BRIEF.md").write_text("the brief")

    def killed(d, prompt):
        (d / "candidate.py").write_text("half written")
        raise KeyboardInterrupt("the step was cancelled")

    with pytest.raises(KeyboardInterrupt):
        bare(killed)._spawn_once(cdir, "go")
    calls = []

    def works(d, prompt):
        calls.append(sorted(p.name for p in d.iterdir()))
        (d / "candidate.py").write_text("a theory")
        return True

    again = bare(works)
    assert again._spawn_once(cdir, "go") is True
    assert calls == [["BRIEF.md"]]  # the killed session's file is not in the new one's directory
    assert (tmp_path / "round_1" / ".killed_candidate_1_1" / "candidate.py").read_text() == "half written"
    assert bare(lambda d, p: pytest.fail("ran twice"))._spawn_once(cdir, "go") is True
