"""Fewest features heuristic listener.

Listeners interpret referring expressions using a non-Bayesian fewest-features
heuristic rather than recursive Theory of Mind. When hearing a word that applies
to multiple referents, listeners select among matching candidates by penalizing
additional unmentioned visual features, favoring the referent with the fewest
total features. On uninformative trials without an informative word, choices
default to a uniform distribution across available objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_fewest_features[u: UTT, r: OBJ](beta, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * vec(feature_count, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L_fewest_features(params["beta"], ctx.lex, ctx.feature_count)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    probs = jnp.where(ctx.is_prior > 0, uniform, heard)
    return with_lapse(probs, params["lapse"])
