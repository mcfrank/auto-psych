"""Feature contrast listener with proportional unmentioned feature penalty.

Listeners interpret referring expressions through direct feature contrast, evaluating
matching referents by penalizing unmentioned features proportionally to each referent's
total feature complexity rather than with a constant additive penalty. The prior over
referents reflects familiarization base rates without an intrinsic feature-count bias,
and prior-elicitation trials reflect this same prior. A lapse parameter mixes in
uniform guessing.
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
    unmentioned_count = jnp.maximum(0.0, ctx.feature_count[None, :] - ctx.lex)
    unmentioned = unmentioned_count / jnp.maximum(1.0, ctx.feature_count[None, :])
    heard = L_contrast(params["theta"], ctx.lex, unmentioned, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
