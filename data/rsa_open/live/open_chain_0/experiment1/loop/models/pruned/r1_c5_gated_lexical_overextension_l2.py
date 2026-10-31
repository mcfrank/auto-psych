"""Refining r1_c5 with positive-feature-gated perceptual distinctiveness and lexical-competition overextension."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
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
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
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
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask
    n_obj = ctx.lex.shape[1]

    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    pos_gate = 1.0 - 0.25 * jnp.exp(-ctx.feature_count)
    distinct = pos_gate * jnp.sum(1.0 - jnp.exp(-total_dist), axis=1)

    effective_w_features = (
        params["w_features"] + ctx.valence * params["w_valence"]
    )

    prior = softmax_prior(
        params["w_distinct"] * distinct
        + effective_w_features * ctx.feature_count
    )

    fc_excess = jnp.maximum(ctx.feature_count - 1.0, 0.0)[None, :]
    true_avail = 1.0 / (1.0 + 0.5 * fc_excess)

    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    overext = jnp.maximum(feat_ext - 2.0, 0.0)
    false_avail = (0.015 + 0.18 * overext) / n_obj

    real_avail = ctx.lex * true_avail + (1.0 - ctx.lex) * false_avail
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * real_avail

    heard = L2(params["alpha"], avail, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
