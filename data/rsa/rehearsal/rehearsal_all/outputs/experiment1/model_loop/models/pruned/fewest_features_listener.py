"""Fewest features heuristic listener.

Listeners interpret referring expressions using a fewest-features parsimony heuristic
rather than recursive pragmatic reasoning. When hearing a word that applies to multiple
candidate referents, listeners select the simplest matching object possessing the minimal
number of total features, penalizing referents with extraneous unmentioned features.
On uninformative trials with no informative word, listeners default to uniform choice
across candidate objects. A lapse parameter captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_fewest[u: UTT, r: OBJ](beta, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * vec(feature_count, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L_fewest(params["beta"], ctx.lex, ctx.feature_count)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
