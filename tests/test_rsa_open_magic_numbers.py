"""No hand-set numbers in a theory (src/rsa_open/loop/gates.py, magic_number_problems; PI 2026-11-08):
outside PARAMS only 0, 1, memo_kit's EPS and BIG, and integers that lay out arrays."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEAD = '''import jax.numpy as jnp
import numpyro.distributions as dist
from src.rsa.memo_kit import BIG, EPS

PARAMS = {"alpha": dist.LogNormal(0.0, 1.0), "w": dist.Normal(0.5, 2.0)}

'''


def problems(body: str):
    from src.rsa_open.loop.gates import magic_number_problems

    return magic_number_problems(HEAD + body)


@pytest.mark.parametrize("body", [
    "def listener(params, ctx):\n    return jnp.where(ctx.is_prior > 0, 1.0, 0.0) * ctx.lex[0]\n",
    "def listener(params, ctx):\n    n = ctx.lex.shape[1]\n    return jnp.ones(n) / n\n",
    "def listener(params, ctx):\n    d = jnp.sum(jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :]), axis=0)\n"
    "    return jnp.min(d + jnp.eye(ctx.lex.shape[1]) * BIG, axis=1)\n",
    "MODES = jnp.arange(2)\n\ndef listener(params, ctx):\n    return ctx.lex.reshape((2, -1))[1]\n",
    "def listener(params, ctx):\n    return jnp.log(ctx.lex[2, 0:3] + EPS) * params['w'] - 1\n",
])
def test_structure_zero_one_and_the_named_constants_pass(body):
    assert problems(body) == []


@pytest.mark.parametrize("body, shows", [
    ("def listener(params, ctx):\n    return 0.87 * ctx.feature_count\n", "0.87"),
    ("def listener(params, ctx):\n    return jnp.where(ctx.feature_count == 4.0, params['w'], 0.0)\n", "== 4.0"),
    ("def listener(params, ctx):\n    n = ctx.lex.shape[1]\n    return jnp.clip(n - 3, 0, 1) * ctx.lex[0]\n", "n - 3"),
    ("def listener(params, ctx):\n    return ctx.feature_count ** 2\n", "** 2"),
    ("def listener(params, ctx):\n    return jnp.log(ctx.lex[0] + 1e-9)\n", "1e-9"),
    ("SHARE = 0.16\n\ndef listener(params, ctx):\n    return SHARE * ctx.lex[0]\n", "SHARE = 0.16"),
    ("def listener(params, ctx, lean=0.3):\n    return lean * ctx.lex[0]\n", "lean=0.3"),
])
def test_weights_thresholds_exponents_and_epsilons_are_refused(body, shows):
    found = problems(body)
    assert len(found) == 1 and shows in found[0] and "put it in PARAMS" in found[0]


def test_the_priors_may_use_any_number():
    assert problems("def listener(params, ctx):\n    return ctx.lex[0] * params['w']\n") == []


def test_every_seed_passes_and_open_chain_1_s_winners_do_not():
    from src.rsa_open.loop.gates import magic_number_problems

    seeds = sorted((ROOT / "data/rsa_open/seeds").glob("*/*.py"))
    assert seeds and not [s.name for s in seeds if magic_number_problems(s.read_text())]
    for e in (1, 2):
        winner = ROOT / f"data/rsa_open/live/open_chain_0/experiment{e}/loop/best_model.py"
        assert magic_number_problems(winner.read_text())


def test_admission_refuses_hand_set_numbers_before_any_fit(tmp_path, monkeypatch):
    from src.rsa_open.loop import gates

    cand = tmp_path / "cand"
    cand.mkdir()
    seed = (ROOT / "data/rsa_open/seeds/bar/rsa_l1.py").read_text()
    (cand / "candidate.py").write_text(seed.replace('heard = L1(params["alpha"], ctx.lex)[ctx.utterance]',
                                                    'heard = L1(0.9 * params["alpha"], ctx.lex)[ctx.utterance]'))
    (cand / "hypothesis.md").write_text("A slightly less rational speaker.")
    monkeypatch.setattr(gates, "loop_fit", lambda *a, **k: pytest.fail("fitted a theory the gate refuses"))
    out = gates.admit(cand, "c", cfg=None, blocks=[], pool=[], admitted_preds={})
    assert not out.admitted and "by hand" in out.reason and "0.9" in out.reason
