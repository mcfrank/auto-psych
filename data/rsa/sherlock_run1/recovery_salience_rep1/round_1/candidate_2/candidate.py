"""Grammatical exhaustivity listener.

Listeners interpret an utterance by applying a closed-world exhaustivity
heuristic: hearing a feature asserts that feature and penalizes candidate
referents in proportion to the number of unmentioned features they possess.
On simple implicature displays this singles out the minimal target, but on
complex displays with symmetric unmentioned features it predicts an even split.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "penalty": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_exh[u: UTT, r: OBJ](penalty, lex: ..., unmentioned: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-penalty * at(unmentioned, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    unmentioned = (
        jnp.maximum(ctx.feature_count[None, :] - 1.0, 0.0)
        * (1.0 - ctx.is_sink[:, None])
    )
    heard = L_exh(params["penalty"], ctx.lex, unmentioned)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
