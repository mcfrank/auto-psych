"""Fewest-features heuristic listener.

Listeners resolve referring expressions using a non-Bayesian simplicity heuristic
rather than recursive pragmatic reasoning about speaker alternatives. Among
candidate objects that literally match the uttered word, listeners favor the
referent possessing the fewest visual features. When no informative word is
spoken, listeners choose uniformly among all objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heuristic[u: UTT, r: OBJ](beta, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * exp(-beta * vec(feature_count, r)))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L_heuristic(params["beta"], ctx.lex, ctx.feature_count)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
