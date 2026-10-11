"""Referential codability listener: pragmatic listener with a communicative affordance prior.

Listeners interpret referring expressions and anticipate speaker intent by expecting speakers
to talk about objects with high referential codability—objects that possess distinguishing
descriptive features that make them viable communicative targets in the scene context.
Before hearing any informative word, listeners evaluate each referent by its communicative
affordance, defined as the total discriminative capacity of the expressions that describe it.
When an utterance is heard, listeners invert an informative speaker using this codability prior.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "w_codable": dist.Normal(1.0, 1.0),
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
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * L0[u, r](lex)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * L1[u, r](alpha, lex, prior)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_codability(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    k_u = jnp.sum(real_lex, axis=1, keepdims=True)
    L0_ext = real_lex / jnp.maximum(1.0, k_u)
    return jnp.sum(L0_ext, axis=0)


def choice_probs(params, ctx):
    codability = compute_codability(ctx)
    prior = softmax_prior(params["w_codable"] * codability)
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
