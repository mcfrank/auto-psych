"""The corpus fit's sampler reaches the log-density through a callback into one separately
compiled value-and-gradient (src/rsa_open/corpus.py, _callback_potential; 2026-11-08), not
inlined into numpyro's compiled sampler. Same density, same sampler, same outputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.rsa.fit import FitSettings
from tests.test_rsa_open_corpus import blocks, theory  # noqa: F401 (fixtures)


def test_the_callback_gives_the_density_and_its_gradient_alone_and_batched(theory, blocks):  # noqa: F811
    from numpyro.infer.util import initialize_model

    from src.rsa_open.corpus import _callback_potential, _corpus_model

    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    info, potential_fn, _, _ = initialize_model(keys, _corpus_model, model_args=(theory, list(blocks)))
    z = info.z
    one = jax.tree_util.tree_map(lambda x: x[1], z)
    potential = _callback_potential(potential_fn, one)
    v, g = jax.value_and_grad(potential_fn)(one)
    cv, cg = jax.jit(jax.value_and_grad(potential))(one)
    assert float(cv) == pytest.approx(float(v), rel=1e-6)
    for k in g:
        np.testing.assert_allclose(np.asarray(cg[k]), np.asarray(g[k]), rtol=1e-5, atol=1e-6)
    bv, bg = jax.jit(jax.vmap(jax.value_and_grad(potential)))(z)  # as vectorized chains call it
    ref = [jax.value_and_grad(potential_fn)(jax.tree_util.tree_map(lambda x: x[i], z)) for i in range(3)]
    np.testing.assert_allclose(np.asarray(bv), [float(r[0]) for r in ref], rtol=1e-6)


def test_a_fit_through_the_callback_returns_constrained_draws_of_every_parameter(theory, blocks):  # noqa: F811
    from src.rsa_open.corpus import fit_corpus

    fit = fit_corpus(theory, blocks, FitSettings(num_warmup=40, num_samples=40, num_chains=2, dense_mass=True))
    post = fit.idata.posterior
    assert set(post.data_vars) == set(fit.param_names)
    assert dict(post.sizes)["chain"] == 2 and dict(post.sizes)["draw"] == 40
    assert float(post["alpha"].min()) > 0 and 0 < float(post["lapse@E"].min()) and float(post["lapse@E"].max()) < 1
    assert fit.pointwise("E").shape == (80, blocks[0].n) and np.all(np.isfinite(fit.pointwise("E")))
    assert fit.timing["warmup_steps"] > 0 and fit.timing["sampling_steps"] > 0


@pytest.mark.slow  # two fits of the same corpus, the callback and numpyro's inlined density
def test_the_callback_and_the_inlined_density_give_the_same_posterior(theory, blocks):  # noqa: F811
    from src.rsa_open.corpus import fit_corpus

    s = FitSettings(num_warmup=400, num_samples=400, num_chains=2, dense_mass=True)
    a = fit_corpus(theory, blocks, s)
    b = fit_corpus(theory, blocks, s, inline_potential=True)
    assert a.converged and b.converged
    for k in a.param_names:
        x, y = np.asarray(a.idata.posterior[k]).ravel(), np.asarray(b.idata.posterior[k]).ravel()
        se = np.sqrt(x.var() / 100 + y.var() / 100)  # a few hundred effective draws each
        assert abs(x.mean() - y.mean()) < 4 * se + 1e-6, (k, x.mean(), y.mean(), se)
