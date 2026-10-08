"""Communicative referability pragmatic listener.

Listeners expect speakers to refer to objects that can be successfully
communicated. When told that a speaker intends to refer to an object using a
single word, the listener does not assume all objects are equally likely
topics of conversation; instead, they form prior expectations over referents
governed by each object's communicative referability—the maximum discriminability
(literal listener accuracy) achievable by any single word for that object in context.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "referability_weight": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, prior: ..., lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    word_sums = jnp.sum(real_lex, axis=1, keepdims=True)
    denom = jnp.where(word_sums == 0.0, 1.0, word_sums)
    l0 = real_lex / denom
    max_discrim = jnp.max(l0 * (1.0 - ctx.is_sink)[:, None], axis=0)

    prior = softmax_prior(params["referability_weight"] * max_discrim)
    heard = L1(params["alpha"], prior, ctx.lex)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
