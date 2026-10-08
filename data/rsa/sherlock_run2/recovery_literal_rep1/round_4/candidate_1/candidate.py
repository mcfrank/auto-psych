"""Aspect goal listener: pragmatic listener reasoning about a speaker informative about object aspects.

Speakers communicate an aspect (feature) of the object rather than its exact
referential identity, choosing which aspect to convey according to its contextual
distinctiveness. A boundedly rational pragmatic listener inverts this speaker by
jointly reasoning about the speaker's communicative aspect goal and intended referent.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](beta, prior: ..., goal_weight: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(q in UTT, wpp=at(goal_weight, q, r)),
        speaker: chooses(u in UTT, to_be=q),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )

    ext = jnp.sum(ctx.lex, axis=1)
    is_real = 1.0 - ctx.is_sink
    distinct = is_real * (1.0 / jnp.maximum(1.0, ext))
    aspect_weight = jnp.exp(params["w_distinct"] * distinct)
    goal_weight = ctx.lex * aspect_weight[:, None]

    heard = L1(params["beta"], prior, goal_weight)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
