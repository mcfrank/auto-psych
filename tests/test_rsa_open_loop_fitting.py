"""Cached, time-limited corpus fits for the open loop (src/rsa_open/loop/fitting.py)."""

import json

import numpy as np
import pytest

from src.rsa.fit import FitSettings
from tests.test_rsa_open_corpus import simulated_block
from tests.test_rsa_open_design import SEEDS

QUICK = FitSettings(num_warmup=40, num_samples=40, num_chains=1, seed=0, dense_mass=True)


@pytest.fixture(scope="module")
def theory_path(tmp_path_factory):
    from src.rsa_open.port import port

    path = tmp_path_factory.mktemp("theory") / "rsa_l1.py"
    path.write_text(port((SEEDS / "rsa_l1.py").read_text(), name="rsa_l1.py"))
    return path


@pytest.fixture(scope="module")
def corpus(theory_path, tmp_path_factory):
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.links import likelihood
    from src.rsa_open.loop.fitting import write_corpus

    theory = TheoryModel(theory_path)
    blocks = [simulated_block(theory, "E", likelihood("choice"), 60, 1, {"temperature": 1.0, "lapse": 0.05}),
              simulated_block(theory, "exp1", likelihood("production"), 60, 2, {"temperature": 1.5, "lapse": 0.05})]
    return blocks, write_corpus(tmp_path_factory.mktemp("corpus") / "corpus.json", blocks)


def test_a_corpus_file_reads_back_as_the_same_blocks(corpus):
    from src.rsa_open.loop.fitting import read_corpus, write_corpus

    blocks, path = corpus
    back = read_corpus(path)
    assert [b.name for b in back] == ["E", "exp1"]
    for a, b in zip(blocks, back):
        assert a.contexts == b.contexts and a.responses == b.responses and a.units == b.units
        assert (a.link.name, a.link.options) == (b.link.name, b.link.options)
    write_corpus(path, blocks)  # the same corpus again: fine
    with pytest.raises(FileExistsError, match="another corpus"):
        write_corpus(path, blocks[:1])


def test_a_fit_is_cached_by_theory_corpus_and_settings(theory_path, corpus, tmp_path):
    from src.rsa_open.loop.fitting import fit_cached

    _, path = corpus
    first = fit_cached(theory_path, "rsa_l1", path, QUICK, tmp_path)
    assert len(list(tmp_path.glob("*.nc"))) == 1
    again = fit_cached(theory_path, "rsa_l1", path, QUICK, tmp_path)
    np.testing.assert_array_equal(first.flat()["alpha"], again.flat()["alpha"])
    assert again.offsets == {"E": (0, 60), "exp1": (60, 120)} and again.blocks == ["E", "exp1"]
    assert {"temperature@E", "temperature@exp1", "lapse@exp1"} <= set(again.flat())


def test_a_theory_whose_code_raises_is_a_model_failure_in_the_child(corpus, tmp_path):
    from src.rsa.loop.fitting import ModelFailure
    from src.rsa_open.loop.fitting import fit_cached
    from src.rsa_open.port import port

    broken = tmp_path / "broken.py"
    broken.write_text(port((SEEDS / "rsa_l1.py").read_text(), name="rsa_l1.py").replace(
        "def listener(params, ctx):", "def listener(params, ctx):\n    raise NameError('undefined_thing')"))
    _, path = corpus
    with pytest.raises(ModelFailure, match="undefined_thing"):
        fit_cached(broken, "broken", path, QUICK, tmp_path / "cache", time_limit_sec=600)


def test_a_fit_past_its_time_limit_is_killed(theory_path, corpus, tmp_path):
    from src.rsa.loop.fitting import FitTimeLimitExceeded
    from src.rsa_open.loop.fitting import fit_cached

    _, path = corpus
    with pytest.raises(FitTimeLimitExceeded):
        fit_cached(theory_path, "rsa_l1", path, FitSettings(num_warmup=2000, num_samples=2000, num_chains=2),
                   tmp_path, time_limit_sec=3)
    assert not list(tmp_path.glob("*.nc"))


def test_a_warm_started_fit_is_keyed_by_its_start_and_a_cold_fit_s_key_is_unchanged(tmp_path):
    from src.rsa.fit import FitSettings
    from src.rsa_open.loop.fitting import fingerprint

    theory, corpus = tmp_path / "t.py", tmp_path / "c.json"
    theory.write_text("x")
    corpus.write_text("y")
    s = FitSettings(dense_mass=True)
    cold = fingerprint(theory, corpus, s)
    assert fingerprint(theory, corpus, s, None) == cold == fingerprint(theory, corpus, s, {})
    warm = fingerprint(theory, corpus, s, {"alpha": 1.4})
    assert warm != cold and warm != fingerprint(theory, corpus, s, {"alpha": 1.5})


def test_a_warm_start_reaches_the_sampler_and_the_cache(theory_path, corpus, tmp_path, monkeypatch):
    import src.rsa_open.corpus as corpus_mod
    from src.rsa_open.loop.fitting import fingerprint, fit_cached
    from src.rsa.loop.fitting import cache_paths

    blocks, path = corpus
    seen = []
    real = corpus_mod.NUTS

    def recording(model, **kw):
        seen.append(kw.get("init_strategy"))
        return real(model, **kw)

    monkeypatch.setattr(corpus_mod, "NUTS", recording)
    start = {"alpha": 1.4, "lapse@E": 0.05, "temperature@exp1": 1.5, "lapse@exp1": 0.05}
    fitted = fit_cached(theory_path, "rsa_l1", path, QUICK, tmp_path, init=start)  # in-process: no time limit
    assert "init_to_value" in repr(seen[-1]) and "1.4" in repr(seen[-1])
    nc, meta = cache_paths(tmp_path, "rsa_l1", fingerprint(theory_path, path, QUICK, start))
    assert nc.exists() and meta.exists() and fitted.idata.posterior["alpha"].size == 40


def test_the_self_check_fits_a_fixed_fifth_of_a_large_block_s_displays():
    from src.rsa_open.corpus import Block
    from src.rsa_open.loop.brief import CHECK_FRACTION, self_check_blocks

    small = simulated_block(_theory(), "exp1", _choice(), 60, 2, {"temperature": 1.0, "lapse": 0.05})
    big0 = simulated_block(_theory(), "E", _choice(), 6000, 1, {"temperature": 1.0, "lapse": 0.05}, units_per=5)
    big = Block("E", big0.link, big0.contexts, big0.responses, big0.units, fixed=(("temperature", 1.0),))
    out = self_check_blocks([big, small])
    assert out[1] is small and out[0].fixed_params == {"temperature": 1.0}
    kept = set(out[0].units)
    assert len(kept) == round(CHECK_FRACTION * len(set(big.units)))
    assert out[0].n == sum(1 for u in big.units if u in kept)  # whole displays
    assert [b.units for b in self_check_blocks([big, small])] == [b.units for b in out]  # fixed


def _theory():
    import tempfile
    from pathlib import Path

    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.port import port

    path = Path(tempfile.mkdtemp()) / "rsa_l1.py"
    path.write_text(port((SEEDS / "rsa_l1.py").read_text(), name="rsa_l1.py"))
    return TheoryModel(path)


def _choice():
    from src.rsa_open.links import likelihood

    return likelihood("choice")


def test_a_fit_child_records_its_peak_memory_and_a_light_read_leaves_out_the_log_likelihood(theory_path, corpus,
                                                                                           tmp_path):
    from src.rsa.loop.fitting import cache_paths
    from src.rsa_open.loop.fitting import fingerprint, fit_cached, scheduler

    _, path = corpus
    light = fit_cached(theory_path, "rsa_l1", path, QUICK, tmp_path, time_limit_sec=600, log_likelihood=False)
    assert "log_likelihood" not in light.idata.groups() and light.idata.posterior["alpha"].size == 40
    _, meta = cache_paths(tmp_path, "rsa_l1", fingerprint(theory_path, path, QUICK))
    peak_mb = json.loads(meta.read_text())["peak_hwm_mb"]
    assert peak_mb > 100  # a child that imported JAX and fitted
    assert scheduler().estimate("rsa_l1") >= peak_mb * 2**20  # the next fits of the theory are given its room
    full = fit_cached(theory_path, "rsa_l1", path, QUICK, tmp_path)  # the cached file keeps it
    assert full.pointwise("exp1").shape == (40, 60)
    np.testing.assert_array_equal(full.flat()["alpha"], light.flat()["alpha"])


def _self_killing(theory_path, tmp_path, times):
    """The theory, killed by SIGKILL (as the job's out-of-memory killer does) on its first ``times`` loads."""
    count = str(tmp_path / "loads")
    path = tmp_path / "killed.py"
    path.write_text(theory_path.read_text() + (
        "\n\nimport os as _os, signal as _signal\n"
        f"_loads = sum(1 for _ in open({count!r})) if _os.path.exists({count!r}) else 0\n"
        f"with open({count!r}, 'a') as _f:\n    _f.write('x\\n')\n"
        f"if _loads < {times}:\n    _os.kill(_os.getpid(), _signal.SIGKILL)\n"))
    return path


def test_a_fit_child_killed_by_the_system_is_fitted_again_once(theory_path, corpus, tmp_path, capsys):
    from src.rsa_open.loop.fitting import fit_cached

    _, path = corpus
    killed_once = _self_killing(theory_path, tmp_path, times=1)
    fitted = fit_cached(killed_once, "killed_once", path, QUICK, tmp_path / "cache", time_limit_sec=600)
    assert fitted.idata.posterior["alpha"].size == 40
    assert "fitting it again" in capsys.readouterr().out


def test_a_fit_child_killed_twice_is_an_infrastructure_failure(theory_path, corpus, tmp_path):
    from src.rsa.loop.fitting import ModelFailure
    from src.rsa_open.loop.fitting import fit_cached

    _, path = corpus
    always = _self_killing(theory_path, tmp_path, times=99)
    with pytest.raises(RuntimeError, match="killed .* twice") as err:
        fit_cached(always, "always_killed", path, QUICK, tmp_path / "cache", time_limit_sec=600)
    assert not isinstance(err.value, ModelFailure)


def test_a_fit_child_s_peak_is_its_own_not_the_size_of_the_process_that_spawned_it(theory_path, corpus, tmp_path):
    from src.rsa.loop.fitting import cache_paths
    from src.rsa_open.loop.fitting import fingerprint, fit_cached, own_rss

    _, path = corpus
    ballast = bytearray(6 * 2**30)  # 6 GB this process really holds when it spawns the child
    ballast[::4096] = b"x" * len(ballast[::4096])
    if own_rss() < 5 * 2**30:
        pytest.skip("no /proc to read this process's memory from")
    fit_cached(theory_path, "rsa_l1", path, QUICK, tmp_path, time_limit_sec=600)
    _, meta = cache_paths(tmp_path, "rsa_l1", fingerprint(theory_path, path, QUICK))
    assert 100 < json.loads(meta.read_text())["peak_hwm_mb"] < 5 * 1024
    del ballast


def test_a_peak_recorded_by_the_inflating_measure_is_ignored(theory_path, corpus, tmp_path):
    from src.rsa.loop.fitting import cache_paths
    from src.rsa_open.loop.fitting import DEFAULT_FIT_BYTES, fingerprint, fit_cached, scheduler

    _, path = corpus
    fit_cached(theory_path, "old_record", path, QUICK, tmp_path)  # in-process: records no peak
    _, meta = cache_paths(tmp_path, "old_record", fingerprint(theory_path, path, QUICK))
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "peak_rss_mb": 15799}))
    fit_cached(theory_path, "old_record", path, QUICK, tmp_path)
    assert scheduler().estimate("old_record") == DEFAULT_FIT_BYTES


# --------------------------------------------------------------------------- where fits run (DISTRIBUTED_DESIGN.md v2)

def test_every_fit_records_the_machine_it_ran_on(tmp_path, monkeypatch):
    import arviz as az

    from src.rsa_open.corpus import CorpusFit
    from src.rsa_open.loop.fitting import _write, machine

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "owners")
    m = machine()
    assert m["slurm_job"] == "123" and m["partition"] == "owners" and m["host"] and m["cpu"]
    fitted = CorpusFit("t", ["E"], az.from_dict(posterior={"a": np.zeros((1, 2))}), ["a"], [])
    _write(fitted, tmp_path / "t.nc", tmp_path / "t.json")
    assert json.loads((tmp_path / "t.json").read_text())["machine"] == m


def test_the_fit_backend_is_local_unless_named_and_never_falls_back(monkeypatch):
    from src.rsa_open.loop.fitting import FIT_BACKEND_ENV, LocalBackend, fit_backend

    monkeypatch.delenv(FIT_BACKEND_ENV, raising=False)
    assert isinstance(fit_backend(), LocalBackend)
    monkeypatch.setenv(FIT_BACKEND_ENV, "queue")
    with pytest.raises(NotImplementedError, match="planned, not built"):
        fit_backend()
    monkeypatch.setenv(FIT_BACKEND_ENV, "elsewhere")
    with pytest.raises(ValueError, match="not a fit backend"):
        fit_backend()


def test_a_killed_attempt_is_retried_once_with_twice_the_room_through_any_backend(tmp_path, monkeypatch):
    import src.rsa_open.loop.fitting as fitting
    from src.rsa.fit import FitSettings
    from src.rsa_open.loop.fitting import FitBackend, ModelFailure, _fit_in_child, _Killed

    class Flaky(FitBackend):
        def __init__(self, kills):
            self.kills, self.calls = kills, []

        def attempt(self, job, at_least=0.0):
            self.calls.append(at_least)
            if len(self.calls) <= self.kills:
                raise _Killed(-9, 4 * 2**30, reserved=10 * 2**30)
            return ("model", "ValueError: the theory's own", "tb")

    args = (tmp_path / "t.py", "flaky_theory", tmp_path / "c.json", FitSettings(), tmp_path / "f.nc",
            tmp_path / "f.json", 60.0)
    once = Flaky(1)
    monkeypatch.setattr(fitting, "fit_backend", lambda: once)
    with pytest.raises(ModelFailure, match="the theory's own"):
        _fit_in_child(*args)
    assert once.calls == [0.0, 20 * 2**30]  # max(2 x 10 GB reserved, 1.25 x the 4 GB peak)
    twice = Flaky(2)
    monkeypatch.setattr(fitting, "fit_backend", lambda: twice)
    with pytest.raises(RuntimeError, match="killed .* twice .* 10 GB reserved"):
        _fit_in_child(*args)


def test_a_fit_s_summary_comes_from_its_sidecar_and_a_cached_refit_is_the_one_used(tmp_path):
    """A1: a resumed loop reads each live theory's summary without loading its draws."""
    from src.rsa.loop.fitting import cache_paths, refit_settings
    from src.rsa_open.loop.fitting import FitSummary, fingerprint, loop_fit_summary

    theory, corpus, cache = tmp_path / "t.py", tmp_path / "c.json", tmp_path / "cache"
    theory.write_text("# t\n")
    corpus.write_text("{}")
    cache.mkdir()
    settings = FitSettings(dense_mass=True)
    assert loop_fit_summary(theory, "t", corpus, settings, cache) is None  # not fitted

    def sidecar(s, problems, warmup):
        _, meta = cache_paths(cache, "t", fingerprint(theory, corpus, s))
        meta.write_text(json.dumps({"convergence_problems": problems,
                                    "timing": {"warmup_sec": warmup, "sampling_sec": 1.0}}))

    sidecar(settings, ["R-hat 1.08"], 10.0)
    assert loop_fit_summary(theory, "t", corpus, settings, cache) == FitSummary(False, ["R-hat 1.08"],
                                                                                {"warmup_sec": 10.0, "sampling_sec": 1.0})
    sidecar(refit_settings(settings), [], 20.0)  # the near miss was refitted, and converged
    assert loop_fit_summary(theory, "t", corpus, settings, cache) == FitSummary(True, [],
                                                                                {"warmup_sec": 20.0, "sampling_sec": 1.0})
