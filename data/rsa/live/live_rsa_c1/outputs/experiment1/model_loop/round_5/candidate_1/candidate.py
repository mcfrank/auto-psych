"""Pragmatic listener inverting a speaker subject to competitive feature interference.

In communicative reference, candidate descriptors for a visually complex referent
compete for lexical retrieval under feature crowding, where mutual inhibitory
interference scales with the referent's total visual feature load. Descriptors that
are contextually primed by appearing across multiple objects in the display resist
this competitive inhibition, whereas idiosyncratic features unique to an individual
object suffer retrieval interference as feature load grows. A rational pragmatic
listener inverts this interference-burdened speaker, recognizing that a speaker
describing a multi-featured referent frequently produces an accessible shared
descriptor rather than retrieving crowded idiosyncratic alternatives.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "lambda_crowd": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., util: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * at(util, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
        + params["w_familiar"] * ctx.familiarization
    )

    l0 = L0(ctx.lex)

    # Contextual occurrence count of each utterance across objects in the display
    occ = jnp.sum(real_lex, axis=1)
    # Idiosyncratic descriptors apply to exactly one object in the display
    is_idiosyncratic = jnp.where(occ == 1.0, 1.0, 0.0) * (1.0 - ctx.is_sink)
    # Feature crowding load: number of competing features on the referent (feature_count - 1)
    crowd_load = jnp.maximum(ctx.feature_count - 1.0, 0.0)
    # Interference cost applies to idiosyncratic descriptors as feature load grows
    cost = params["lambda_crowd"] * (is_idiosyncratic[:, None] * crowd_load[None, :])
    util = jnp.log(l0 + EPS) - cost

    heard = L1(params["alpha"], ctx.lex, prior, util)[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
