"""Step mode decides exactly as a local run (DISTRIBUTED_DESIGN.md v3 §0b; the infrastructure
amendment rests on it). The open inner loop runs as a sequence of steps: each step runs until a
phase has fits pending, submits them (to a stand-in Slurm that records the arrays) with its
continuation, and ends; the arrays' tasks then run here (`fit_array.run_task`, as on Sherlock),
and a fresh loop object (a new job) continues from disk. Against a loop run straight through with
the same theorist, it must give the same ledger, history and export, call each agent once, and
never fit in a step's own process."""

import json
from pathlib import Path

import pytest

from tests.test_rsa_open_loop_orchestrator import SETTINGS, FakeTheorist, _inputs


class LocalSlurm:
    """sbatch records an array's manifest; the arrays run after the step ends (`run_arrays`)."""

    def __init__(self):
        self.manifests, self.next, self.calls = [], 100, []

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        if cmd[0] == "squeue":
            return ""
        if cmd[0] == "sbatch":
            self.next += 1
            self.manifests.append(Path(cmd[-1]))
            return f"{self.next}\n"
        if cmd[0] == "bash":  # the continuation
            self.next += 1
            return f"{self.next}\n"
        return ""

    def run_arrays(self) -> int:
        """Each task as `fit_array.sbatch` runs it: its own process, with none of the step's settings
        (Slurm's ``--export=NONE``), so it fits with the local backend. In this process the test has
        set the slurm backend and made the local one fail (review V3 §11: a `fold_after` task run here
        found its fold pending)."""
        import os
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if not k.startswith("RSA_OPEN_")}
        n = 0
        for m in self.manifests:
            for i in range(len(json.loads(m.read_text())["tasks"])):
                done = subprocess.run([sys.executable, "-m", "src.rsa_open.loop.fit_array", "--manifest", str(m),
                                       "--index", str(i)], env=env, cwd=Path(__file__).resolve().parents[1],
                                      capture_output=True, text=True)
                assert done.returncode == 0, (m.name, i, done.stdout[-2000:], done.stderr[-4000:])
                n += 1
        self.manifests.clear()
        return n


def step_config(cell: Path):
    from src.rsa_open.loop.slurm import StepConfig

    cell.mkdir(parents=True, exist_ok=True)
    return StepConfig(cell, "rsa_open_test", cell / "continuation.sh", cell / "fit_array.sbatch",
                      {"VENV_PY": "/v", "HARNESS_REPO": "/h", "JAX_COMPILATION_CACHE_DIR": "/j"}, cell / "logs")


def run_in_steps(cfg, theorist, cell: Path, max_steps: int = 60):
    """The loop as a chain of steps (what run_chain does in a step job). Returns (final, steps)."""
    import src.rsa_open.loop.slurm as step
    from src.rsa_open.chain import clear_stage
    from src.rsa_open.loop.fit_array import FitsPending
    from src.rsa_open.loop.orchestrator import OpenLoop

    slurm, steps = LocalSlurm(), 0
    while True:
        stepper = step.Stepper(step_config(cell), slurm)
        step.activate(stepper)
        resume = (cfg.results_dir / "history.json").exists()
        if not resume:
            clear_stage(cfg.results_dir, keep=(".fit_cache", ".scores"))  # as chain.model_loop does in a step
        try:
            return OpenLoop(cfg, theorist).run(resume=resume), steps
        except FitsPending as ps:
            with pytest.raises(step.StepYield):
                stepper.yield_step(ps, "test")
        finally:
            step.activate(None)
        steps += 1
        assert slurm.run_arrays() > 0 or stepper.inflight, "a step yielded with nothing to wait for"
        assert steps < max_steps, "the chain is not converging"


@pytest.mark.slow  # two loops of two rounds, every fit twice (straight, then as array tasks)
def test_a_loop_run_as_steps_decides_exactly_as_one_run_straight_through(tmp_path, monkeypatch):
    from src.rsa_open.loop.fitting import LocalBackend
    from src.rsa_open.loop.orchestrator import OpenLoop, OpenLoopConfig

    def config(root):
        root.mkdir()
        seeds, corpus = _inputs(root)
        return OpenLoopConfig(corpus_path=corpus, seed_models_dir=seeds, results_dir=root / "loop", max_iterations=2,
                              candidate_count=3, settings=SETTINGS, fit_time_limit_sec=1800)
        # A time limit, as every real loop has: unlimited fits went in-process (review V3 §11), and now
        # go to arrays in step mode too, but the straight run's fits should be the children a real one runs.

    straight_theorist = FakeTheorist()
    OpenLoop(config(tmp_path / "a"), straight_theorist).run()

    stepped_cfg = config(tmp_path / "b")
    stepped_theorist = FakeTheorist()

    def never(*a, **k):
        raise AssertionError("a step fitted in its own process")

    monkeypatch.setenv("RSA_OPEN_FIT_BACKEND", "slurm")
    monkeypatch.setattr(LocalBackend, "attempt", never)
    _, steps = run_in_steps(stepped_cfg, stepped_theorist, tmp_path / "cell")
    assert steps >= 3  # the seeds (their folds follow them), and at least one yield a round

    def ledger(root):
        return (root / "loop" / "attempted_hypotheses.jsonl").read_text().splitlines()

    def history(root):
        return [(h["round"], h["best_model"], [(e.get("name"), e.get("outcome")) for e in h["events"]],
                 {n: (round(d["sel_elpd"], 6), d["trusted"], d["eligible"]) for n, d in h["standing"].items()})
                for h in json.loads((root / "loop" / "history.json").read_text())]

    def calls(theorist):
        return sorted(name for name, _ in theorist.calls)

    assert ledger(tmp_path / "a") == ledger(tmp_path / "b")
    assert history(tmp_path / "a") == history(tmp_path / "b")
    assert (json.loads((tmp_path / "a" / "loop" / "export.json").read_text())
            == json.loads((tmp_path / "b" / "loop" / "export.json").read_text()))
    assert calls(straight_theorist) == calls(stepped_theorist)  # every agent session once, none again
    log = [json.loads(line) for line in (tmp_path / "cell" / "steps.jsonl").read_text().splitlines()]
    assert sum(1 for e in log if e.get("continuation")) == steps
    assert not (tmp_path / "cell" / "STOP").exists()


def run_in_local_steps(cfg, theorist, cell: Path, max_steps: int = 60):
    """The loop as run_chain runs it with fits in the step's own job (`LocalStepper`): each phase's fits
    run here as `fit_array` tasks in their own processes, then a fresh loop object continues from disk.
    Returns (final, steps)."""
    import src.rsa_open.loop.slurm as step
    from src.rsa_open.chain import clear_stage
    from src.rsa_open.loop.fit_array import FitsPending
    from src.rsa_open.loop.orchestrator import OpenLoop

    steps = 0
    while True:
        stepper = step.LocalStepper(step_config(cell), workers=2)
        step.activate(stepper)
        resume = (cfg.results_dir / "history.json").exists()
        if not resume:
            clear_stage(cfg.results_dir, keep=(".fit_cache", ".scores"))
        try:
            return OpenLoop(cfg, theorist).run(resume=resume), steps
        except FitsPending as ps:
            with pytest.raises(step.StepContinue) as cont:
                stepper.yield_step(ps, "test")
            assert cont.value.record["failed_tasks"] == 0
        finally:
            step.activate(None)
            stepper.close()
        steps += 1
        assert steps < max_steps, "the chain is not converging"


@pytest.mark.slow  # two loops of two rounds, every fit twice (straight, then in the step's own job)
def test_a_loop_whose_steps_run_their_own_fits_decides_exactly_as_one_run_straight_through(tmp_path, monkeypatch):
    import sys

    from src.rsa_open.loop.fitting import LocalBackend
    from src.rsa_open.loop.orchestrator import OpenLoop, OpenLoopConfig

    def config(root):
        root.mkdir()
        seeds, corpus = _inputs(root)
        return OpenLoopConfig(corpus_path=corpus, seed_models_dir=seeds, results_dir=root / "loop", max_iterations=2,
                              candidate_count=3, settings=SETTINGS, fit_time_limit_sec=1800)

    straight_theorist = FakeTheorist()
    OpenLoop(config(tmp_path / "a"), straight_theorist).run()

    def never(*a, **k):
        raise AssertionError("a step fitted in its own process, not a task's")

    monkeypatch.setenv("RSA_OPEN_FIT_BACKEND", "slurm")
    monkeypatch.setenv("VENV_PY", sys.executable)
    monkeypatch.setenv("HARNESS_REPO", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "jax_cache"))
    monkeypatch.setattr(LocalBackend, "attempt", never)
    stepped_theorist = FakeTheorist()
    _, steps = run_in_local_steps(config(tmp_path / "b"), stepped_theorist, tmp_path / "cell")
    assert steps >= 3

    def ledger(root):
        return (root / "loop" / "attempted_hypotheses.jsonl").read_text().splitlines()

    def history(root):
        return [(h["round"], h["best_model"], [(e.get("name"), e.get("outcome")) for e in h["events"]],
                 {n: (round(d["sel_elpd"], 6), d["trusted"], d["eligible"]) for n, d in h["standing"].items()})
                for h in json.loads((root / "loop" / "history.json").read_text())]

    assert ledger(tmp_path / "a") == ledger(tmp_path / "b")
    assert history(tmp_path / "a") == history(tmp_path / "b")
    assert (json.loads((tmp_path / "a" / "loop" / "export.json").read_text())
            == json.loads((tmp_path / "b" / "loop" / "export.json").read_text()))
    assert sorted(n for n, _ in straight_theorist.calls) == sorted(n for n, _ in stepped_theorist.calls)
    log = [json.loads(line) for line in (tmp_path / "cell" / "steps.jsonl").read_text().splitlines()]
    assert sum(1 for e in log if e.get("local") and not e.get("early")) == steps and not any(e.get("continuation") for e in log)
    assert not (tmp_path / "cell" / "STOP").exists()
