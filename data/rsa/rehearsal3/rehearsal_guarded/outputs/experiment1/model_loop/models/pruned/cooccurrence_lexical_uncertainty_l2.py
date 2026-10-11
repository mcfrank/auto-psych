"""Pragmatic listener reasoning under co-occurrence lexical uncertainty.

Listeners interpret referring expressions under lexical uncertainty about which visual
feature a word designates in context. Lexical interference arises from visual feature
co-occurrence: candidate words and features that co-occur on the same context objects
compete during lexical selection and perception. Ambiguous expressions with rich feature
overlap naturally elevate the perceived plausibility of non-matching distractors, while
displays with isolated, non-overlapping features maintain sharp semantic boundaries.
Choices on uninformative prior trials are guided by visual singleton salience, contextual
distinctiveness, and feature complexity.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "nu": dist.Beta(1.0, 19.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 19.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., channel: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(f in UTT, wpp=at(lex, f, r) * exp(alpha * log(L0[f, r](lex) + {EPS}))),
        speaker: chooses(u in UTT, wpp=at(channel, u, f)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., channel: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            f in UTT,
            wpp=at(lex, f, r) * exp(alpha * log(L1[f, r](alpha, lex, channel, prior) + {EPS})),
        ),
        speaker: chooses(u in UTT, wpp=at(channel, u, f)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_prior(ctx, w_singleton, w_distinct, w_features):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    return softmax_prior(
        w_singleton * is_singleton
        + w_distinct * distinct
        + w_features * ctx.feature_count
    )


def compute_channel(ctx, nu):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    cooccur = real_lex @ real_lex.T
    col_sums = jnp.sum(cooccur, axis=0, keepdims=True)
    norm_cooccur = cooccur / jnp.maximum(col_sums, 1.0)

    eye = jnp.eye(ctx.lex.shape[0])
    # For columns with positive sum, interpolate eye with norm_cooccur.
    # For sink columns (or empty), use eye.
    noisy_channel = (1.0 - nu) * eye + nu * norm_cooccur
    return jnp.where((col_sums > 0) & (ctx.is_sink[None, :] == 0), noisy_channel, eye)


def choice_probs(params, ctx):
    prior = compute_prior(ctx, params["w_singleton"], params["w_distinct"], params["w_features"])
    channel = compute_channel(ctx, params["nu"])
    heard = L2(params["alpha"], ctx.lex, channel, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
