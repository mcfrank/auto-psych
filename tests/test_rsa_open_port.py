"""Porting closed RSA model files to theory + link (docs/auto_open/PLAN.md §3.1)."""

import ast

import numpy as np
import pytest

from src.rsa.context import group_by_shape
from src.rsa.loop.novelty import plain_pool
from src.rsa.model_file import RSAModel, prior_draws
from src.runtime.config import PROJECT_ASSETS_DIR, REPO_ROOT

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
PROMOTED = REPO_ROOT / "data" / "rsa" / "live_seeds" / "models"
FINALS = REPO_ROOT / "data" / "rsa" / "live" / "finals"

# Chain 2's final mixes its lapse with its own prior, not a uniform guess (PLAN.md §3.1).
UNPORTABLE = {"competition_endpoint_binding_lapse_l2"}


@pytest.fixture(scope="module")
def sample():
    pool = plain_pool()
    rng = np.random.default_rng(0)
    return group_by_shape([pool[i] for i in rng.choice(len(pool), 60, replace=False)])


def ported_and_composed(original, tmp_path):
    from src.rsa_open.links import compose
    from src.rsa_open.port import port

    theory = tmp_path / f"{original.stem}.theory.py"
    theory.write_text(port(original.read_text(), name=original.name))
    model = tmp_path / f"{original.stem}.py"
    model.write_text(compose(theory.read_text(), "choice"))
    return theory, model


def same_distribution(a, b):
    return type(a) is type(b) and all(
        np.array_equal(np.asarray(getattr(a, k)), np.asarray(getattr(b, k))) for k in a.arg_constraints)


def assert_same_choice_probs(original, composed, sample, n_draws=3):
    a, b = RSAModel(original), RSAModel(composed)
    assert list(a.params) == list(b.params)  # same names, same order: the same sampler
    for name in a.params:
        assert same_distribution(a.params[name], b.params[name]), name
    for draw in prior_draws(a.params, n_draws, 3):
        for group in sample.values():
            np.testing.assert_array_equal(np.asarray(a.group_probs(draw, group)),
                                          np.asarray(b.group_probs(draw, group)))


def test_a_ported_model_composed_with_the_choice_link_gives_its_choice_probabilities(sample, tmp_path):
    original = SEEDS / "rsa_l1_salience.py"
    theory, composed = ported_and_composed(original, tmp_path)
    tree = ast.parse(theory.read_text())
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "listener" in defined and "choice_probs" not in defined
    assert "lapse" not in RSAModel(theory).params  # the theory alone has no lapse
    assert_same_choice_probs(original, composed, sample)


def test_the_port_leaves_every_other_line_as_it_was():
    from src.rsa_open.port import port

    original = (SEEDS / "rsa_l1_salience.py").read_text()
    ported = port(original, name="rsa_l1_salience.py")
    removed = set(original.splitlines()) - set(ported.splitlines())
    assert removed == {
        '    "lapse": dist.Beta(1.0, 9.0),',
        "def choice_probs(params, ctx):",
        '    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])',
    }


def test_the_port_refuses_a_model_without_a_final_uniform_lapse():
    from src.rsa_open.port import PortRefused, port

    with pytest.raises(PortRefused, match="with_lapse"):
        port((FINALS / "competition_endpoint_binding_lapse_l2.py").read_text(),
             name="competition_endpoint_binding_lapse_l2.py")


def test_the_port_refuses_a_lapse_prior_other_than_the_choice_link_s():
    from src.rsa_open.port import PortRefused, port

    source = (SEEDS / "rsa_l1.py").read_text().replace("dist.Beta(1.0, 9.0)", "dist.Beta(1.0, 4.0)")
    with pytest.raises(PortRefused, match="Beta"):
        port(source, name="rsa_l1.py")


def test_the_port_refuses_a_lapse_used_anywhere_but_the_final_mixture():
    from src.rsa_open.port import PortRefused, port

    source = (SEEDS / "rsa_l1.py").read_text().replace(
        "def choice_probs(params, ctx):", 'def choice_probs(params, ctx):\n    _ = params["lapse"] * 2.0')
    with pytest.raises(PortRefused, match="lapse"):
        port(source, name="rsa_l1.py")


def test_a_theory_may_not_name_a_link_parameter():
    from src.rsa_open.links import LinkClash, compose

    theory = (SEEDS / "rsa_l1.py").read_text().replace("def choice_probs", "def listener")
    with pytest.raises(LinkClash, match="lapse"):
        compose(theory, "choice")


def test_composing_with_an_unknown_link_fails_loudly():
    from src.rsa_open.links import compose

    with pytest.raises(KeyError, match="no link"):
        compose("PARAMS = {}\n", "telepathy")


@pytest.mark.slow  # compiles every seed and final twice
@pytest.mark.parametrize("original", sorted(
    [*SEEDS.glob("*.py"), *PROMOTED.glob("*.py"), *FINALS.glob("*.py")], key=lambda p: str(p)),
    ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_portable_seed_and_final_reproduces_its_choice_probabilities(original, sample, tmp_path):
    from src.rsa_open.port import PortRefused

    if original.stem in UNPORTABLE:
        with pytest.raises(PortRefused):
            ported_and_composed(original, tmp_path)
        return
    _, composed = ported_and_composed(original, tmp_path)
    assert_same_choice_probs(original, composed, sample)
