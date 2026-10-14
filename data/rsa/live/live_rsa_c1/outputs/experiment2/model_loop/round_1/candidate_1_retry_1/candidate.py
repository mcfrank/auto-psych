"""Pragmatic listener inverting a speaker with pragmatic preemption.

Speakers prioritize unambiguous, uniquely discriminating descriptors that uniquely identify
an intended referent. When an unambiguous descriptor is available for an intended referent,
using an ambiguous, shared descriptor is pragmatically preempted. A rational listener inverts
this preemption-sensitive speaker: hearing a shared descriptor, the listener infers that
the speaker intended a referent that lacked an unambiguous distinguishing label rather than
a competitor that could have been singled out by an unambiguous descriptor.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "p_preempt": dist.Beta(1.0, 2.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_max_feat": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., factor: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * at(factor, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
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

    max_feat_count = jnp.max(ctx.feature_count)
    is_max_feat = jnp.where(
        (ctx.feature_count == max_feat_count) & (ctx.feature_count > 0.0),
        1.0,
        0.0,
    )

    prior = softmax_prior(
        params["w_singleton"] * featured_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_max_feat"] * is_max_feat
    )

    ext_size = jnp.sum(real_lex, axis=1)
    is_unique_utt = jnp.where(ext_size == 1.0, 1.0, 0.0) * (1.0 - ctx.is_sink)
    is_shared_utt = jnp.where(ext_size > 1.0, 1.0, 0.0) * (1.0 - ctx.is_sink)

    has_unique = jnp.where(
        jnp.sum(real_lex * is_unique_utt[:, None], axis=0) > 0.0,
        1.0,
        0.0,
    )

    preempt_discount = jnp.clip(1.0 - params["p_preempt"], a_min=1e-4, a_max=1.0)
    preempt_mask = is_shared_utt[:, None] * has_unique[None, :]
    factor = 1.0 - (1.0 - preempt_discount) * preempt_mask

    heard = L1(params["alpha"], ctx.lex, prior, factor)[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
