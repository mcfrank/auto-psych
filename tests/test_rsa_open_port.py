"""Porting closed RSA model files to theory + link (docs/auto_open/PLAN.md §3.1)."""

import ast
import hashlib

import numpy as np
import pytest

from src.rsa.context import group_by_shape
from src.rsa.loop.novelty import plain_pool
from src.rsa.model_file import RSAModel, prior_draws
from src.runtime.config import PROJECT_ASSETS_DIR, REPO_ROOT

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
PROMOTED = REPO_ROOT / "data" / "rsa" / "live_seeds" / "models"
FINALS = REPO_ROOT / "data" / "rsa" / "live" / "finals"
ALL_MODELS = sorted([*SEEDS.glob("*.py"), *PROMOTED.glob("*.py"), *FINALS.glob("*.py")], key=str)
CHAIN2 = FINALS / "competition_endpoint_binding_lapse_l2.py"
SOFTMAX = PROMOTED / "softmax_surprisal_isolated_costly_l4.py"


@pytest.fixture(scope="module")
def sample():
    pool = plain_pool()
    rng = np.random.default_rng(0)
    return group_by_shape([pool[i] for i in rng.choice(len(pool), 60, replace=False)])


def ported_and_composed(original, tmp_path):
    from src.rsa_open.links import compose
    from src.rsa_open.port import port_file

    theory = tmp_path / f"{original.stem}.theory.py"
    theory.write_text(port_file(original))
    model = tmp_path / f"{original.stem}.py"
    model.write_text(compose(theory.read_text(), "choice"))
    return theory, model


def same_distribution(a, b):
    return type(a) is type(b) and all(
        np.array_equal(np.asarray(getattr(a, k)), np.asarray(getattr(b, k))) for k in a.arg_constraints)


def composed_params(draw, rule):
    """The composed model's parameters equivalent to the original's ``draw``."""
    if rule == "a":  # the theory's lapse is the link's; temperature 1
        return {**draw, "temperature": 1.0}
    if rule == "b":  # the theory keeps its own mixture; the link adds nothing
        out = {("theory_lapse" if k == "lapse" else k): v for k, v in draw.items()}
        return {**out, "lapse": 0.0, "temperature": 1.0}
    raise ValueError(rule)


def theory_names(original_params, rule):
    """The ported theory's parameter names, in order, from the original's."""
    names = list(original_params)
    if rule == "a":
        return [k for k in names if k != "lapse"]
    if rule == "b":
        return ["theory_lapse" if k == "lapse" else k for k in names]
    return [k for k in names if k not in ("lapse", "beta")]  # the hand port


def assert_same_choice_probs(original, theory, composed, sample, n_draws=3):
    from src.rsa_open.port import port_rule

    rule = port_rule(theory.read_text())
    a, b = RSAModel(original), RSAModel(composed)
    assert list(b.params) == [*theory_names(a.params, rule), "lapse", "temperature"]
    renamed = {"theory_lapse": "lapse"} if rule == "b" else {}
    for name in theory_names(a.params, rule):
        assert same_distribution(b.params[name], a.params[renamed.get(name, name)]), name
    if rule in ("a", "hand"):
        assert same_distribution(b.params["lapse"], a.params["lapse"])
    for draw in prior_draws(a.params, n_draws, 3):
        if rule == "hand":
            draw = {**draw, "beta": 1.0}
            ours = {**{k: v for k, v in draw.items() if k != "beta"}, "temperature": 1.0}
        else:
            ours = composed_params(draw, rule)
        for group in sample.values():
            np.testing.assert_allclose(np.asarray(b.group_probs(ours, group)),
                                       np.asarray(a.group_probs(draw, group)), rtol=1e-5, atol=1e-6)


def test_a_ported_model_composed_with_the_choice_link_gives_its_choice_probabilities(sample, tmp_path):
    original = SEEDS / "rsa_l1_salience.py"
    theory, composed = ported_and_composed(original, tmp_path)
    tree = ast.parse(theory.read_text())
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "listener" in defined and "choice_probs" not in defined
    assert "lapse" not in RSAModel(theory).params  # the theory alone has no lapse
    assert list(RSAModel(composed).params)[-2:] == ["lapse", "temperature"]  # the link's, appended
    assert_same_choice_probs(original, theory, composed, sample)


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


def test_a_model_without_a_final_lapse_is_ported_whole_with_its_lapse_renamed():
    from src.rsa_open.port import port, port_rule

    ported = port(CHAIN2.read_text(), name=CHAIN2.name)
    assert port_rule(ported) == "b"
    assert 'params["theory_lapse"]' in ported and '"theory_lapse": dist.Beta' in ported
    assert not any(isinstance(n, ast.Constant) and n.value == "lapse" for n in ast.walk(ast.parse(ported)))
    removed = set(CHAIN2.read_text().splitlines()) - set(ported.splitlines())
    assert all("lapse" in line or "def choice_probs" in line for line in removed)


def test_a_model_with_some_returns_lapsed_and_some_not_is_refused():
    from src.rsa_open.port import PortRefused, port

    source = (SEEDS / "rsa_l1.py").read_text().replace(
        "def choice_probs(params, ctx):", "def choice_probs(params, ctx):\n    if False:\n        return 0.0")
    with pytest.raises(PortRefused, match="neither rule"):
        port(source, name="rsa_l1.py")


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


def test_the_softmax_seed_is_ported_by_hand_without_its_temperature():
    from src.rsa_open.port import HAND_PORTS, port_file, port_rule

    sha, _ = HAND_PORTS[SOFTMAX.name]
    assert hashlib.sha256(SOFTMAX.read_bytes()).hexdigest() == sha
    theory = port_file(SOFTMAX)
    assert port_rule(theory) == "hand"
    tree = ast.parse(theory)
    assert not any(isinstance(n, ast.Constant) and n.value in ("beta", "lapse") for n in ast.walk(tree))


def test_a_changed_original_of_a_hand_port_is_refused(tmp_path):
    from src.rsa_open.port import PortRefused, port_file

    changed = tmp_path / SOFTMAX.name
    changed.write_text(SOFTMAX.read_text() + "\n# edited\n")
    with pytest.raises(PortRefused, match="redo the hand port"):
        port_file(changed)


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
@pytest.mark.parametrize("original", ALL_MODELS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_seed_and_final_ports_and_reproduces_its_choice_probabilities(original, sample, tmp_path):
    theory, composed = ported_and_composed(original, tmp_path)
    assert_same_choice_probs(original, theory, composed, sample)


@pytest.mark.slow  # compiles the depth-4 seed
def test_the_hand_port_equals_its_original_on_word_trials_at_temperature_beta(sample, tmp_path):
    theory, composed = ported_and_composed(SOFTMAX, tmp_path)
    a, b = RSAModel(SOFTMAX), RSAModel(composed)
    for draw in prior_draws(a.params, 3, 5):
        ours = {**{k: v for k, v in draw.items() if k != "beta"}, "temperature": draw["beta"]}
        for group in sample.values():
            words = np.asarray(group.arrays.is_prior) == 0
            np.testing.assert_allclose(np.asarray(b.group_probs(ours, group))[words],
                                       np.asarray(a.group_probs(draw, group))[words], rtol=2e-4, atol=1e-6)
