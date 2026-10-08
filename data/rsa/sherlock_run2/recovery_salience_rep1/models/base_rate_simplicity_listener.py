"""RSA pragmatic listener (depth 1) with feature simplicity and familiarization base-rate tracking.

Refinement of feature_simplicity_listener: Communicators integrate empirical base
rates established during familiarization with an inductive feature-simplicity prior
over referents (taken from base_rate_tracking_listener).
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "simplicity": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    simplicity_prior = softmax_prior(params["simplicity"] * ctx.feature_count)
    uniform = jnp.full_like(ctx.familiarization, 1.0 / ctx.familiarization.shape[-1])
    raw_base_rate = jnp.where(ctx.has_familiarization > 0, ctx.familiarization, uniform)
    combined = raw_base_rate * simplicity_prior
    prior = (combined + EPS) / jnp.sum(combined + EPS)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
