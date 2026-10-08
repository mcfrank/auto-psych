"""RSA pragmatic listener reasoning about a speaker with aspect-directed communicative goals.

Rather than assuming the speaker's goal is to identify the full object token,
the listener simulates a speaker whose goal is to convey an aspect (feature) of the
referent. The speaker chooses an aspect true of the referent and selects an utterance
to maximize the literal listener's belief that the referent possesses that aspect.
The pragmatic listener inverts this generative model, jointly inferring the referent
and the speaker's communicative goal.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def AspectBelief[u: UTT, g: UTT](lex: ..., prior: ...):
    listener: knows(u, g)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[at(lex, g, listener.r) == 1]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            g in UTT,
            u in UTT,
            wpp=at(lex, g, r)
            * at(lex, u, r)
            * exp(alpha * log(AspectBelief[u, g](lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
