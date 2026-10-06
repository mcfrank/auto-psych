"""Loading memo model files per context shape, and the model contract."""

import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from src.rsa.context import Context, group_by_shape
from src.rsa.model_file import ModelContractViolation, RSAModel, check_contract
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
SEED_NAMES = [
    "literal_listener",
    "rsa_l1",
    "rsa_l2",
    "rsa_l1_salience",
    "rsa_l1_shared_prior",
]

SIMPLE = dict(objects=((0, 0), (0, 1), (1, 1)), feature_names=("hat", "glasses"))
# The pragmods "complex" game: objects M, GM, HG over hat, glasses, mustache.
COMPLEX = dict(
    objects=((0, 0, 1), (0, 1, 1), (1, 1, 0)), feature_names=("hat", "glasses", "mustache")
)


def probs(model, params, contexts):
    groups = group_by_shape(contexts)
    out = np.zeros((len(contexts), max(c.shape[0] for c in contexts)))
    for group in groups.values():
        p = np.asarray(model.group_probs(params, group))
        out[group.indices, : p.shape[1]] = p
    return out


def test_literal_and_pragmatic_listeners_give_the_textbook_values():
    glasses = Context(**SIMPLE, utterance=1)
    l0 = RSAModel(SEEDS / "literal_listener.py")
    np.testing.assert_allclose(probs(l0, {"lapse": 0.0}, [glasses])[0], [0, 0.5, 0.5], atol=1e-6)
    l1 = RSAModel(SEEDS / "rsa_l1.py")
    np.testing.assert_allclose(
        probs(l1, {"alpha": 1.0, "lapse": 0.0}, [glasses])[0], [0, 0.75, 0.25], atol=1e-6
    )


def test_depth_two_favours_the_twice_implicated_object_more_than_depth_one():
    glasses = Context(**COMPLEX, utterance=1)
    params = {"alpha": 1.0, "lapse": 0.0}
    p1 = probs(RSAModel(SEEDS / "rsa_l1.py"), params, [glasses])[0]
    p2 = probs(RSAModel(SEEDS / "rsa_l2.py"), params, [glasses])[0]
    assert p2[1] > p1[1] > 0.5


def test_one_file_serves_contexts_of_different_shapes():
    model = RSAModel(SEEDS / "rsa_l1.py")
    contexts = [
        Context(**SIMPLE, utterance=1),
        Context(objects=((1, 0), (1, 1), (0, 1), (1, 0)), feature_names=("a", "b"), utterance=0),
    ]
    p = probs(model, {"alpha": 2.0, "lapse": 0.1}, contexts)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert len(model.compiled_shapes) == 2


def test_prior_query_uses_the_salience_prior():
    model = RSAModel(SEEDS / "rsa_l1_salience.py")
    ctx = Context(**SIMPLE, utterance=None)
    p = probs(model, {"alpha": 1.0, "w_features": 1.0, "w_familiar": 0.0, "lapse": 0.0}, [ctx])[0]
    expected = np.exp([0.0, 1.0, 2.0]) / np.exp([0.0, 1.0, 2.0]).sum()
    np.testing.assert_allclose(p, expected, atol=1e-5)


@pytest.mark.parametrize("name", SEED_NAMES)
def test_every_seed_meets_the_contract(name):
    contexts = [
        Context(**SIMPLE, utterance=1),
        Context(**SIMPLE, utterance=None, familiarization=(0.1, 0.1, 0.8)),
        Context(**COMPLEX, utterance=2),
    ]
    check_contract(RSAModel(SEEDS / f"{name}.py"), group_by_shape(contexts))


GOOD_BODY = """
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo
from src.rsa.memo_kit import at, with_lapse

PARAMS = {"lapse": dist.Beta(1.0, 9.0)}

@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]

def choice_probs(params, ctx):
    p = L0(ctx.lex)[ctx.utterance]
    return %s
"""


def write_model(tmp_path, body, name="m"):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(body))
    return RSAModel(path)


def groups():
    return group_by_shape([Context(**SIMPLE, utterance=1)])


def test_a_well_formed_model_passes(tmp_path):
    check_contract(write_model(tmp_path, GOOD_BODY % 'with_lapse(p, params["lapse"])'), groups())


@pytest.mark.parametrize(
    "returned, message",
    [
        ("p[:2]", "shape"),
        ("2.0 * p", "sum to 1"),
        ("p + 0.6 * (jnp.arange(3) == 0) - 0.6 * (jnp.arange(3) == 2)", "negative"),
        ("p * jnp.nan", "finite"),
        ('jnp.sqrt(params["lapse"] - params["lapse"]) + p', "gradient"),
    ],
)
def test_contract_violations_raise(tmp_path, returned, message):
    with pytest.raises(ModelContractViolation, match=message):
        check_contract(write_model(tmp_path, GOOD_BODY % returned), groups())


def test_a_model_may_not_define_the_injected_domains(tmp_path):
    body = GOOD_BODY.replace("PARAMS =", "OBJ = jnp.arange(3)\nPARAMS =") % "p"
    with pytest.raises(ModelContractViolation, match="OBJ"):
        write_model(tmp_path, body)


def test_a_model_needs_params_and_choice_probs(tmp_path):
    with pytest.raises(ModelContractViolation, match="PARAMS"):
        check_contract(
            write_model(tmp_path, (GOOD_BODY % "p").replace("PARAMS =", "_P =")), groups()
        )
    with pytest.raises(ModelContractViolation, match="choice_probs"):
        check_contract(
            write_model(tmp_path, (GOOD_BODY % "p").replace("def choice_probs", "def other")),
            groups(),
        )
