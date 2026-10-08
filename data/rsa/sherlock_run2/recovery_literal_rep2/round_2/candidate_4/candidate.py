"""Feature contrast listener with neutral prior: direct feature contrast without feature-count prior bias.

Listeners interpret referring expressions by directly evaluating matching referents
according to feature contrast, penalizing additional unmentioned features. The
prior over candidate referents is neutral with respect to feature count, reflecting
only familiarization base rates when available. A lapse parameter mixes in uniform
guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "theta": dist.LogNormal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_contrast[u: UTT, r: OBJ](theta, lex: ..., unmentioned: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r) * exp(-theta * at(unmentioned, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(params["w_familiar"] * ctx.familiarization)
    unmentioned = jnp.maximum(0.0, ctx.feature_count[None, :] - ctx.lex)
    heard = L_contrast(params["theta"], ctx.lex, unmentioned, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
