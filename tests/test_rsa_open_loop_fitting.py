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
