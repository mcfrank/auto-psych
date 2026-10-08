"""Grammatical exhaustification listener.

Listeners resolve reference through grammatical exhaustification rather than
recursive theory-of-mind reasoning. A heard feature word is interpreted as an
exhaustive description of the intended referent, penalizing candidate objects
that possess extraneous, unmentioned features by exp(-beta * unmentioned).
With no informative word the choice is uniform. A lapse parameter mixes in
uniform guessing.
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
def L_exh[u: UTT, r: OBJ](beta, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * (vec(feature_count, r) - 1.0)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L_exh(params["beta"], ctx.lex, ctx.feature_count)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
