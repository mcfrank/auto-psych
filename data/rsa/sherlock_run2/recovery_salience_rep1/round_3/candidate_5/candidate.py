"""RSA pragmatic listener (depth 1) combining familiarization base rates, feature simplicity, and selective visual attention.

Refinement of base_rate_simplicity_listener: Listeners integrate prior expectations
(familiarization base rates and feature simplicity) with selective visual attention
to candidate referents (taken from selective_attention_listener).
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "distractor_attention": dist.Beta(1.0, 1.0),
    "simplicity": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., att: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(att, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., att: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, att) + {EPS}))),
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
    matching = ctx.lex[ctx.utterance]
    att = jnp.where(matching > 0, 1.0, params["distractor_attention"])
    heard = L1(params["alpha"], ctx.lex, att, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
