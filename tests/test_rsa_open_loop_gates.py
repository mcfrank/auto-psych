"""Admission of theories to the open loop (src/rsa_open/loop/gates.py)."""

import numpy as np
import pytest

from src.rsa.context import Context
from tests.test_rsa_open_design import SEEDS
from src.rsa.fit import FitSettings

# Two chains: the convergence gate needs R-hat.
GATE_SETTINGS = FitSettings(num_warmup=150, num_samples=150, num_chains=2, seed=0, dense_mass=True)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from src.rsa.loop.novelty import plain_pool
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.links import likelihood
    from src.rsa_open.loop.fitting import write_corpus
    from src.rsa_open.loop.gates import GateConfig
    from src.rsa_open.port import port
    from tests.test_rsa_open_corpus import simulated_block

    root = tmp_path_factory.mktemp("gates")
    seed = root / "rsa_l1.py"
    seed.write_text(port((SEEDS / "rsa_l1.py").read_text(), name="rsa_l1.py"))
    theory = TheoryModel(seed)
    blocks = [simulated_block(theory, "E", likelihood("choice"), 60, 1, {"temperature": 1.0, "lapse": 0.05}),
              simulated_block(theory, "exp1", likelihood("binary_judgment"), 60, 2,
                              {"temperature": 1.0, "lapse": 0.05, "center": 0.0})]
    corpus = write_corpus(root / "corpus.json", blocks)
    pool = [c for c in plain_pool() if c.shape in ((2, 2), (3, 3))][:40]
    cfg = GateConfig(corpus, root / "cache", GATE_SETTINGS, novelty_threshold=0.002, time_limit_sec=None)
    return root, seed, blocks, pool, cfg


def candidate(root, name, source, hypothesis="A hypothesis."):
    d = root / name
    d.mkdir(exist_ok=True)
    (d / "candidate.py").write_text(source)
    (d / "hypothesis.md").write_text(hypothesis)
    return d


def admit(setup, d, admitted=None):
    from src.rsa_open.loop.gates import admit

    _, _, blocks, pool, cfg = setup
    return admit(d, d.name, cfg=cfg, blocks=blocks, pool=pool, admitted_preds=admitted or {})


def test_a_ported_theory_is_admitted_and_a_copy_of_it_is_not_novel(setup):
    root, seed, *_ = setup
    first = admit(setup, candidate(root, "pragmatic", seed.read_text()))
    assert first.admitted, first.reason
    assert first.fit.blocks == ["E", "exp1"] and np.all(np.isfinite(first.pool_preds))
    again = admit(setup, candidate(root, "pragmatic_copy", seed.read_text()), {"pragmatic": first.pool_preds})
    assert not again.admitted and again.reason.startswith("not novel") and "pragmatic" in again.reason


@pytest.mark.parametrize("edit, reason", [
    (lambda s: s.replace("def listener(params, ctx):", "def choice_probs(params, ctx):"), "not a theory"),
    (lambda s: s.replace('"alpha"', '"temperature"'), "belong to the links"),
    (lambda s: "import os\n" + s, "code gate"),
    (lambda s: s + "\n\ndef speaker(params, ctx):\n    return jnp.ones_like(ctx.lex)\n", "speaker"),
])
def test_files_that_are_not_sound_theories_are_refused_with_the_reason(setup, edit, reason):
    root, seed, *_ = setup
    result = admit(setup, candidate(root, f"bad_{abs(hash(reason)) % 997}", edit(seed.read_text())))
    assert not result.admitted and reason in result.reason, result.reason


def test_an_incomplete_candidate_is_refused(setup):
    root, seed, *_ = setup
    d = candidate(root, "no_hypothesis", seed.read_text(), hypothesis="")
    assert "hypothesis.md" in admit(setup, d).reason


def test_the_derived_speaker_meets_the_speaker_contract(setup):
    from src.rsa.context import group_by_shape
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.loop.gates import check_speaker_contract

    _, seed, _, pool, _ = setup
    check_speaker_contract(TheoryModel(seed), group_by_shape(pool))
    sink_scene = Context(objects=((1, 0), (1, 1), (0, 0)), feature_names=("a", "b"), utterance=0)
    check_speaker_contract(TheoryModel(seed), group_by_shape([sink_scene]))
