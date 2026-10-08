"""Non-Bayesian heuristic listener selecting referents by perceptual isolation.

Rather than recursively simulating counterfactual utterances via Bayes' rule, the
listener employs a direct perceptual isolation heuristic: among candidate objects
that literally match the heard word, the listener chooses with probability weighted
by the object's nearest-neighbor visual contrast (perceptual isolation), favoring
isolated singleton referents that pop out from context distractors. On uninformative
prior trials without an informative word, choice directly reflects this perceptual
isolation distribution. A lapse parameter mixes in random choice.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def object_isolation(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Minimum Hamming distance of each object to any other object in the display."""
    features = lex * (1.0 - is_sink)[:, None]
    diff = jnp.abs(features[:, :, None] - features[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    n_obj = lex.shape[1]
    eye = jnp.eye(n_obj) * 1e5
    return jnp.min(pair_dist + eye, axis=1)


@memo
def H[u: UTT, r: OBJ](beta, lex: ..., isolation: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(beta * vec(isolation, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    iso_vec = object_isolation(ctx.lex, ctx.is_sink)
    prior = softmax_prior(params["beta"] * iso_vec)
    heard = H(params["beta"], ctx.lex, iso_vec)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
