"""Individuated exemplar contrast Theory of Mind with complementary figure-ground salience and clone-sensitive minimal-exemplar grounding."""

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
    fc = ctx.feature_count

    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = (
        jnp.sum(diff, axis=0)
        + jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
        + jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    )
    mean_dist = jnp.sum(pair_dist, axis=1) / denom
    dup_count = jnp.sum(pair_dist == 0.0, axis=1)
    is_singleton = jnp.where(dup_count == 1.0, 1.0, 0.0)

    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    unique_fc = jnp.sum(real_lex * jnp.where(feat_ext == 1.0, 1.0, 0.0) * real_mask, axis=0)
    f3_mask = jnp.where((feat_ext == 3.0) & (n_obj == 4), 1.0, 0.0) * real_mask
    has_f3 = jnp.any(f3_mask > 0.0)
    maj_abs = jnp.sum((1.0 - real_lex) * f3_mask, axis=0)

    plain_gate = 1.5 / denom + 0.55 * jnp.maximum(mean_dist - 2.0, 0.0)
    pos_gate = jnp.where(
        fc == 0,
        plain_gate,
        1.0 - 0.25 * jnp.exp(-fc),
    )
    contrast = (0.88 / jnp.sqrt(denom)) * pos_gate * jnp.sum(1.0 - jnp.exp(-pair_dist), axis=1)

    has_univ4 = jnp.any(jnp.where((feat_ext == 4.0) & (n_obj == 4), 1.0, 0.0) * real_mask > 0.0)
    var_mask = jnp.where((feat_ext > 0.0) & (feat_ext < n_obj), 1.0, 0.0) * real_mask
    var_fc = jnp.sum(real_lex * var_mask, axis=0)

    salience = (
        contrast
        + (0.22 + 0.50 * ctx.valence) * fc
        + 0.52 * jnp.where(fc == 1.0, 1.0, 0.0) * unique_fc * maj_abs
        + 0.32 * has_univ4 * (1.0 - has_f3) * jnp.maximum(var_fc - 2.0, 0.0)
        + 0.30 * (1.0 - ctx.grayscale)
        + 0.60 * ctx.familiarization
    )
    prior = softmax_prior(params["w_salience"] * salience)

    # Speaker lexical retrieval: narrow diagnostic features (ext <= 2) compete for selection.
    diag_w = jnp.where(feat_ext <= 2.0, 1.0, 0.0) * real_mask
    diag_fc = jnp.sum(real_lex * diag_w, axis=0, keepdims=True)
    fc_excess = jnp.maximum(diag_fc - 1.0, 0.0)
    true_avail = 1.0 / (1.0 + 0.50 * fc_excess)
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * (ctx.lex * true_avail)

    prag = L2(params["alpha"], avail, prior)[ctx.utterance]

    # Individuated minimal-exemplar grounding on over-extended utterances (heard_ext >= 3).
    heard_row = real_lex[ctx.utterance]
    heard_ext = jnp.sum(heard_row)
    overext = jnp.maximum(heard_ext - 2.0, 0.0)
    is_overext = jnp.clip(overext, 0.0, 1.0)
    univ_crowd = jnp.maximum(heard_ext - 3.0, 0.0)

    min_match_fc = jnp.min(jnp.where(heard_row > 0.0, fc, 100.0))
    is_min_match = jnp.where((heard_row > 0.0) & (fc == min_match_fc), 1.0, 0.0)
    n_min_match = jnp.maximum(jnp.sum(is_min_match), 1.0)
    min_match_cloned = jnp.where(jnp.sum(is_min_match * (1.0 - is_singleton)) > 0.0, 1.0, 0.0)

    heard_unique = heard_row * jnp.where(heard_ext == 1.0, 1.0, 0.0)
    has_unique_alt = jnp.clip(unique_fc - heard_unique, 0.0, 1.0)

    rel_fc = jnp.clip(fc - min_match_fc, 0.0, 1.0) * (1.0 - 0.65 * min_match_cloned) / n_min_match
    lit_excess = 0.50 * jnp.maximum(diag_fc[0] - 0.8, 0.0) * (1.0 - has_unique_alt) + 1.80 * rel_fc
    indiv_mult = (
        1.0
        + (1.80 * (1.0 - univ_crowd) + 0.45 * univ_crowd) * is_min_match * has_unique_alt
        + 0.70 * min_match_cloned * (1.0 - is_min_match) * is_singleton * has_unique_alt
    )
    lit_w = heard_row * indiv_mult / (1.0 + 0.65 * univ_crowd * lit_excess)
    lit0 = lit_w / jnp.maximum(jnp.sum(lit_w), EPS)

    ret_l0 = (0.16 + 0.08 * jnp.clip(n_obj - heard_ext, 0.0, 1.0)) * is_overext + 0.30 * univ_crowd
    ret_prior = 0.0032 * overext
    heard = (1.0 - ret_l0 - ret_prior) * prag + ret_l0 * lit0 + ret_prior * prior

    return jnp.where(ctx.is_prior > 0, prior, heard)
