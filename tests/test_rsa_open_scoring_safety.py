"""Run C′'s scoring failure (PI 2026-10-28): an unconverged fold of `rsa_l2` scored +121,409 nats on a
bet paradigm, and every other theory was then compared against it.

- the bet link's Dirichlet-multinomial lost its precision at large concentrations in float32 (at
  1e8, +224 for a log probability whose value is -4.3): `links.log_rising` keeps it exact;
- a held-out log probability above 0 is impossible for any link: the theory is untrusted;
- only converged cross-validation is compared.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from src.rsa_open.corpus import CorpusCV


@pytest.mark.parametrize("kappa", [1e-3, 1.0, 9.99, 10.0, 1e3, 1e5, 1e8, 1e10])
@pytest.mark.parametrize("total", [10, 100, 1000])
def test_the_bet_likelihood_is_the_dirichlet_multinomial_at_every_concentration(kappa, total):
    from src.rsa_open.link_kit import temper, with_uniform
    from src.rsa_open.links import allocation_log_prob

    rng = np.random.default_rng(int(kappa) % 97 + total)
    latent = rng.dirichlet(np.ones(4), size=40).astype(np.float32)
    points = np.array([rng.multinomial(total, p) for p in latent], dtype=np.float32)
    ours = np.asarray(allocation_log_prob(jnp.asarray(latent), jnp.asarray(points), total, kappa, 0.05))
    with jax.experimental.enable_x64():
        conc = np.maximum(kappa * np.asarray(with_uniform(temper(jnp.asarray(latent, jnp.float64), 1.0), 0.05)), 1e-8)
        ref = np.asarray(dist.DirichletMultinomial(jnp.asarray(conc), total_count=total).log_prob(
            jnp.asarray(points, jnp.float64)))
    assert np.all(ours <= 1e-3)  # a log probability
    assert np.abs(ours - ref).max() < 5e-3


def test_numpyro_s_own_form_was_the_failure():
    conc = 1e8 * jnp.array([0.7, 0.2, 0.1], dtype=jnp.float32)
    assert float(dist.DirichletMultinomial(conc, total_count=100).log_prob(jnp.array([70.0, 20.0, 10.0]))) > 0


def test_the_bet_likelihood_has_finite_gradients_across_the_switch():
    from src.rsa_open.links import allocation_log_prob

    lat = jnp.array([[0.6, 0.3, 0.1]], dtype=jnp.float32)
    pts = jnp.array([[50.0, 30.0, 20.0]])
    g = jax.grad(lambda k: allocation_log_prob(lat, pts, 100, k, 0.05).sum())
    assert all(np.isfinite(float(g(k))) for k in (0.01, 9.99, 10.0, 10.01, 1e6))


def _loop(tmp_path, pointwise, converged):
    from src.rsa_open.loop.orchestrator import OpenLoop, OpenLoopConfig

    loop = OpenLoop.__new__(OpenLoop)
    loop.cfg = OpenLoopConfig(corpus_path=tmp_path, seed_models_dir=tmp_path, results_dir=tmp_path)
    n = 30
    loop.blocks = [SimpleNamespace(name="E", units=tuple(range(n)), n=n),
                   SimpleNamespace(name="exp1", units=tuple(range(n)), n=n)]
    loop.cv = {k: CorpusCV({"E": e, "exp1": x}, converged[k]) for k, (e, x) in pointwise.items()}
    loop.cv_problems = {}
    loop.live_cfg = SimpleNamespace(time_limit_sec=None)
    loop.live = {k: SimpleNamespace(fit=SimpleNamespace(converged=True)) for k in pointwise}
    return loop


def test_only_converged_cross_validation_is_compared(tmp_path):
    n = 30
    rng = np.random.default_rng(0)
    pointwise = {"good": (-1 + rng.normal(0, 0.1, n), -1 + rng.normal(0, 0.1, n)),
                 "other": (-1.2 + rng.normal(0, 0.1, n), -1.5 + rng.normal(0, 0.1, n)),
                 "spike": (-1.0 + rng.normal(0, 0.1, n), np.full(n, 4000.0))}  # run C′'s degenerate fold
    loop = _loop(tmp_path, pointwise, {"good": True, "other": True, "spike": False})
    s = loop.standing()
    assert s["spike"]["sel_elpd"] == -np.inf and not s["spike"]["trusted"]
    assert s["good"]["by_block"]["exp1"]["behind"] == 0  # not 120,000 nats behind the spike
    assert loop.incumbent(s) == "good"


def test_an_impossible_held_out_score_makes_a_theory_untrusted():
    from src.rsa_open.loop.orchestrator import IMPOSSIBLE_LPD

    src = open("src/rsa_open/loop/orchestrator.py").read()
    assert "np.nanmax(v) > IMPOSSIBLE_LPD" in src and IMPOSSIBLE_LPD == 1e-3
    assert "an impossible held-out score" in src


def test_an_unchanged_repair_is_refused_without_a_refit():
    src = open("src/rsa_open/loop/orchestrator.py").read()
    assert 'after.read_bytes() == before.read_bytes()' in src
    assert "the repair left candidate.py unchanged, so the rejection stands" in src


def test_the_critic_is_told_to_keep_its_context_small_and_theorists_not_to_fit_their_own():
    critic = open("src/rsa_open/loop/prompts/open_critique.md").read()
    theorist = open("src/rsa_open/loop/prompts/open_theory.md").read()
    assert "Your context is limited" in critic and "head -n 20" in critic
    assert "Do not fit theories with code of your own" in theorist and "stop" in theorist
