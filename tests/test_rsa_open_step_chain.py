"""A chain run as steps (src/rsa_open/run_chain.py, src/rsa_open/chain.py): the exit codes the
step job reads, the mode's guards, and that re-running a chain from its start each step never
designs, deploys or publishes an experiment again."""

import types

import pytest

from tests.test_rsa_open_live import fake_services, frozen_experiment, live_cfg, records, stored  # noqa: F401
from tests.test_rsa_open_step_submit import FakeSlurm, _cfg, _pending


def _main_with(monkeypatch, tmp_path, run_fn, stepper):
    import src.rsa_open.run_chain as rc

    monkeypatch.setattr(rc, "build", lambda args: types.SimpleNamespace(run=run_fn, cfg=None))
    monkeypatch.setattr(rc, "step_from_env", lambda: stepper)
    monkeypatch.setattr(rc, "this_launch", lambda cfg, chain: {"experiment": 1})
    monkeypatch.setattr(rc, "replace", lambda cfg, **kw: cfg)
    return rc.main(types.SimpleNamespace(run_dir=tmp_path / "run"))


def test_a_step_with_fits_pending_submits_them_and_its_continuation_and_exits_6(tmp_path, monkeypatch):
    from src.rsa_open.loop import slurm as step

    slurm = FakeSlurm()
    stepper = step.Stepper(_cfg(tmp_path), slurm)
    step.activate(stepper)
    pending = _pending(tmp_path)

    def run():
        raise pending

    with pytest.raises(SystemExit) as out:
        _main_with(monkeypatch, tmp_path, run, stepper)
    assert out.value.code == step.EXIT_YIELD == 6
    assert [c[0] for c in slurm.calls] == ["squeue", "sbatch", "sbatch", "bash", "scontrol"]
    assert step.active() is None  # deactivated however the chain ends


def test_a_step_that_may_not_submit_exits_7_with_its_stop_written(tmp_path, monkeypatch):
    from src.rsa_open.loop import slurm as step

    cfg = _cfg(tmp_path, max_queued=2)
    stepper = step.Stepper(cfg, FakeSlurm(queued=1))
    pending = _pending(tmp_path)

    def run():
        raise pending

    with pytest.raises(SystemExit) as out:
        _main_with(monkeypatch, tmp_path, run, stepper)
    assert out.value.code == step.EXIT_REFUSED == 7
    assert "jobs already queued" in (cfg.cell_dir / "STOP").read_text()


def test_fits_pending_outside_step_mode_is_an_error_not_a_yield(tmp_path, monkeypatch):
    pending = _pending(tmp_path)

    def run():
        raise pending

    with pytest.raises(RuntimeError, match="outside step mode"):
        _main_with(monkeypatch, tmp_path, run, None)


@pytest.mark.parametrize("mode,backend", [("1", ""), ("", "slurm"), ("1", "local")])
def test_step_mode_and_the_slurm_fit_backend_go_together(monkeypatch, mode, backend):
    from src.rsa_open.run_chain import step_from_env

    monkeypatch.setenv("RSA_OPEN_STEP_MODE", mode)
    monkeypatch.setenv("RSA_OPEN_FIT_BACKEND", backend)
    with pytest.raises(RuntimeError, match="go together"):
        step_from_env()


def test_a_step_needs_its_settings_from_the_job(monkeypatch):
    from src.rsa_open.run_chain import step_from_env

    monkeypatch.setenv("RSA_OPEN_STEP_MODE", "1")
    monkeypatch.setenv("RSA_OPEN_FIT_BACKEND", "slurm")
    monkeypatch.delenv("RSA_OPEN_CELL_DIR", raising=False)
    with pytest.raises(RuntimeError, match="RSA_OPEN_CELL_DIR"):
        step_from_env()


def test_steps_re_running_a_live_chain_never_deploy_or_publish_an_experiment_again(tmp_path, fake_services,  # noqa: F811
                                                                                  monkeypatch):
    """Every step runs the chain from its start; finished stages are skipped. Here experiment 1's
    loop yields twice (fits pending) after its study was published and collected."""
    from src.rsa_open.chain import Chain, ChainConfig
    from src.rsa_open.loop import slurm as step
    from src.rsa_open.loop.fit_array import FitsPending

    p, lists, run = frozen_experiment(tmp_path / "run")
    L = lists["lists"]
    fake_services["responses"][:] = [stored(f"x{k}", L[k], records(L[k]), at=k) for k in range(5)]
    loops = []

    def model_loop(self, n):
        loops.append(n)
        if len(loops) < 3:
            raise _pending(tmp_path / f"s{len(loops)}")
        return {"best_model": "x", "live": ["x"]}

    monkeypatch.setattr(Chain, "model_loop", model_loop)
    monkeypatch.setattr(Chain, "design_context", lambda self, n: pytest.fail("a frozen design was designed again"))
    slurm = FakeSlurm()
    steps = 0
    while True:
        chain = Chain(ChainConfig(chain="chain_0", arm="fixed", seeds_dir=tmp_path, literature=None, out_dir=run,
                                  ground_truth=None, live=live_cfg(tmp_path), n_experiments=1),
                      designer_spawn=None, theorist_spawn=None)
        stepper = step.Stepper(_cfg(tmp_path) if steps == 0 else stepper.cfg, slurm)
        step.activate(stepper)
        try:
            chain.run()
            break
        except FitsPending as ps:
            with pytest.raises(step.StepYield):
                stepper.yield_step(ps)
        finally:
            step.activate(None)
        steps += 1
    assert steps == 2 and loops == [1, 1, 1]
    assert len(fake_services["deploy"]) == 1  # one page, one study
