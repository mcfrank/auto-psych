"""Mutual exclusivity listener.

Listeners interpret words using the principle of mutual exclusivity rather
than recursive Theory-of-Mind simulation, choosing among matching referents
with a penalty for candidate objects that possess additional competing
features.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse

PARAMS = {
    "gamma": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_me[u: UTT, r: OBJ](gamma, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-gamma * vec(feature_count, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L_me(params["gamma"], ctx.lex, ctx.feature_count)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
