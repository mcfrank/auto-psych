"""Pragmatic listener inverting a speaker under noisy truth-conditional semantics.

Listeners interpret utterances under noisy truth-conditional semantics: words do not
function as rigid boolean filters but have soft, probabilistic applicability across
objects. Even when a descriptor is literally absent from an object, there is a small
baseline semantic compatibility that permits the word to denote non-matching referents.
A rational pragmatic listener inverts a speaker operating under these soft truth values.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "delta": dist.Beta(1.0, 19.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, soft_lex: ..., prior: ..., util: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(soft_lex, u, r) * exp(alpha * at(util, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    duplicate_count = jnp.sum(pair_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)
    featured_singleton = is_singleton * (ctx.feature_count > 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    prior = softmax_prior(
        params["w_singleton"] * featured_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
    )

    # Soft truth-conditional semantics: real features have baseline semantic compatibility delta
    delta = params["delta"]
    soft_lex = jnp.where(
        ctx.is_sink[:, None] > 0.0,
        ctx.lex,
        (1.0 - delta) * ctx.lex + delta,
    )

    # Literal listener over soft semantics
    l0 = soft_lex / (jnp.sum(soft_lex, axis=1, keepdims=True) + EPS)
    util = jnp.log(l0 + EPS)

    heard = L1(params["alpha"], soft_lex, prior, util)[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
