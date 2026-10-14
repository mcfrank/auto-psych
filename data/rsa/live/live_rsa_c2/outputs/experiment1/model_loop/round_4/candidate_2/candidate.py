"""Pragmatic listener at depth 2 reasoning about a speaker with strategic silence.

In reference games, speakers are not compelled to produce an ambiguous utterance
when all available words fail to uniquely identify the referent; rather, they have
an outside option to remain silent. When all candidate labels for an object are
ambiguous, the speaker frequently chooses silence rather than communicating
unreliably, while distinguishing features prompt reliable speech. Depth-2 pragmatic
listeners invert this strategic silence speaker, softening preemption penalties
against competitor objects in scalar contexts.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

ACTIONS = jnp.arange(2)

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_silence": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_empty": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def S1[u: UTT, r: OBJ](alpha, w_silence, lex: ...):
    speaker: knows(r)
    speaker: chooses(
        act in ACTIONS,
        cand in UTT,
        wpp=(1 - act) * at(lex, cand, r) * exp(alpha * log(L0[cand, r](lex) + {EPS}))
        + act * (cand == 0) * exp(w_silence),
    )
    return Pr[speaker.act == 0, speaker.cand == u]


@memo
def L1[u: UTT, r: OBJ](lex: ..., prior: ..., s1: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * vec(prior, r) * at(s1, u, r),
    )
    return Pr[listener.r == r]


@memo
def S2[u: UTT, r: OBJ](alpha, w_silence, lex: ..., l1: ...):
    speaker: knows(r)
    speaker: chooses(
        act in ACTIONS,
        cand in UTT,
        wpp=(1 - act) * at(lex, cand, r) * exp(alpha * log(at(l1, cand, r) + {EPS}))
        + act * (cand == 0) * exp(w_silence),
    )
    return Pr[speaker.act == 0, speaker.cand == u]


@memo
def L2[u: UTT, r: OBJ](lex: ..., prior: ..., s2: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * vec(prior, r) * at(s2, u, r),
    )
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

    is_empty = jnp.where(ctx.feature_count == 0, 1.0, 0.0)

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_empty"] * is_empty
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    s1 = S1(params["alpha"], params["w_silence"], ctx.lex)
    l1 = L1(ctx.lex, prior, s1)
    s2 = S2(params["alpha"], params["w_silence"], ctx.lex, l1)
    heard = L2(ctx.lex, prior, s2)[ctx.utterance]

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
