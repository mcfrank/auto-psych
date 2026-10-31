"""Resource-limited listener with selective display attention and focal working-memory capacity bounds on referential tracking."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_salience": dist.LogNormal(0.0, 0.5),
}


@memo
def L0[u: UTT, r: OBJ](avail: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(avail, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, avail: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(avail, u, r) * exp(alpha * log(L0[u, r](avail) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, avail: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(avail, u, r) * exp(alpha * log(L1[u, r](alpha, avail, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask
    n_obj = ctx.lex.shape[1]
    denom = jnp.maximum(n_obj - 1.0, 1.0)

    # Selective display attention: pairwise perceptual contrast and diagnostic dimension coverage
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = (
        jnp.sum(diff, axis=0)
        + jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
        + jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    )
    mean_dist = jnp.sum(pair_dist, axis=1) / denom
    plain_gate = 1.5 / denom + 0.55 * jnp.maximum(mean_dist - 2.0, 0.0)
    pos_gate = jnp.where(
        ctx.feature_count == 0,
        plain_gate,
        1.0 - 0.25 * jnp.exp(-ctx.feature_count),
    )
    contrast = (0.88 / jnp.sqrt(denom)) * pos_gate * jnp.sum(1.0 - jnp.exp(-pair_dist), axis=1)

    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    var_mask = jnp.where((feat_ext > 0.0) & (feat_ext < n_obj), 1.0, 0.0) * real_mask
    n_var = jnp.maximum(jnp.sum(var_mask), 1.0)
    var_fc = jnp.sum(real_lex * var_mask, axis=0)
    has_univ = jnp.sum(jnp.where(feat_ext == n_obj, 1.0, 0.0) * real_mask)
    proto_share = (var_fc / n_var) ** 2

    salience = (
        contrast
        + (0.17 + 0.50 * ctx.valence) * ctx.feature_count
        + (0.14 + 0.28 * has_univ) * proto_share
        + 0.30 * (1.0 - ctx.grayscale)
        + 0.60 * ctx.familiarization
    )
    prior = softmax_prior(params["w_salience"] * salience)

    # Diagnostic lexical competition during speaker retrieval
    diag_w = jnp.where(feat_ext <= 2.0, 1.0, 0.12) * real_mask
    diag_fc = jnp.sum(real_lex * diag_w, axis=0, keepdims=True)
    fc_excess = jnp.maximum(diag_fc - 1.0, 0.0)
    true_avail = 1.0 / (1.0 + 0.45 * fc_excess)
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * (ctx.lex * true_avail)

    prag = L2(params["alpha"], avail, prior)[ctx.utterance]

    # Focal working-memory capacity bound (>2 referents in crowded >=4-object displays):
    # minimal-anchor category filtering and incomplete attentional suppression of excluded items.
    heard_row = real_lex[ctx.utterance]
    heard_ext = jnp.sum(heard_row)
    overext = jnp.maximum(heard_ext - 2.0, 0.0) * jnp.clip(n_obj - 3.0, 0.0, 1.0)
    univ_crowd = jnp.maximum(heard_ext - 3.0, 0.0)

    min_match_fc = jnp.min(jnp.where(heard_row > 0, ctx.feature_count, 100.0))
    rel_fc = jnp.maximum(ctx.feature_count - min_match_fc, 0.0)
    lit_excess = 0.40 * jnp.maximum(diag_fc[0] - 1.0, 0.0) + 1.80 * (1.0 - jnp.exp(-rel_fc))
    lit_w = heard_row / (1.0 + univ_crowd * lit_excess)
    lit0 = lit_w / jnp.maximum(jnp.sum(lit_w), EPS)

    crowd_load = overext / (1.0 + 0.35 * jnp.maximum(overext - 1.0, 0.0))
    ret_l0 = 0.34 * crowd_load
    ret_prior = 0.013 * overext
    heard = (1.0 - ret_l0 - ret_prior) * prag + ret_l0 * lit0 + ret_prior * prior

    return jnp.where(ctx.is_prior > 0, prior, heard)
