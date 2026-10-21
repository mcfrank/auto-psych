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
