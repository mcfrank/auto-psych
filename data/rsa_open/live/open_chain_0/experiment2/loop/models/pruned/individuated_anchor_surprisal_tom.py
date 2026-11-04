"""Referential individuation and bilateral feature surprisal Theory of Mind."""

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


def _scene_components(params, ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask
    n_obj = ctx.lex.shape[1]
    denom = jnp.maximum(n_obj - 1.0, 1.0)
    fc = ctx.feature_count
    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)

    # Pairwise perceptual dissimilarity and token singleton status (unique vs duplicate clones).
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    feat_dist = jnp.sum(diff, axis=0)
    pair_dist = (
        feat_dist
        + jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
        + jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    )
    dup_count = jnp.sum(jnp.where(pair_dist == 0.0, 1.0, 0.0), axis=1)
    is_singleton = jnp.where(dup_count == 1.0, 1.0, 0.0)

    # Bilateral feature surprisal:
    # 1. Scene-unique positive features (feat_ext == 1) uniquely individuate their bearer.
    is_uniq_feat = jnp.where(feat_ext == 1.0, 1.0, 0.0) * real_mask
    has_uniq = jnp.clip(jnp.sum(real_lex * is_uniq_feat, axis=0), 0.0, 1.0)

    # 2. Scene-varying features vs invariant universal background features.
    var_mask = jnp.where((feat_ext > 0.0) & (feat_ext < n_obj), 1.0, 0.0) * real_mask
    var_fc = jnp.sum(real_lex * var_mask, axis=0)
    has_univ = jnp.max(jnp.where(feat_ext == n_obj, 1.0, 0.0) * real_mask)

    # 3. Majority-exclusion oddball: in 4-object scenes with a 3-object majority feature,
    # a minimal-feature outsider (min_fc < 2) that lacks the majority feature stands out by absence contrast.
    maj_mask = jnp.where((feat_ext == 3.0) & (n_obj == 4), 1.0, 0.0) * real_mask
    lacks_maj = jnp.max((1.0 - real_lex) * maj_mask, axis=0)
    min_fc = jnp.min(fc)
    is_min_oddball = jnp.where((fc == min_fc) & (min_fc < 2.0), 1.0, 0.0) * lacks_maj

    mean_dist = jnp.sum(pair_dist, axis=1) / denom
    neg_val = jnp.maximum(-ctx.valence, 0.0)
    plain_gate = 1.5 / denom + 0.85 * jnp.maximum(mean_dist - 2.0, 0.0) + 0.80 * neg_val
    pos_gate = jnp.where(fc == 0, plain_gate, 1.0 - 0.25 * jnp.exp(-fc))
    contrast = (0.88 / jnp.sqrt(denom)) * pos_gate * jnp.sum(1.0 - jnp.exp(-pair_dist), axis=1)

    pos_singleton = is_singleton * jnp.clip(fc, 0.0, 1.0)
    indiv_sal = 0.26 * pos_singleton * (1.0 + 0.70 * has_uniq) + 0.45 * is_min_oddball

    salience = (
        contrast
        + indiv_sal
        + (0.21 + 0.65 * ctx.valence) * fc
        + 0.24 * has_univ * jnp.maximum(var_fc - 2.0, 0.0)
        + 0.30 * (1.0 - ctx.grayscale)
        + 0.65 * ctx.familiarization
    )
    prior = softmax_prior(params["w_salience"] * salience)

    # Moderated communicative prior inside recursive belief reasoning (avoids compounding
    # background figure completeness and familiarization base rates across L1 and L2).
    comm_sal = (
        contrast
        + 0.20 * pos_singleton * (1.0 + 0.70 * has_uniq)
        + (0.20 + 0.60 * ctx.valence) * fc
        + 0.30 * (1.0 - ctx.grayscale)
        + 0.15 * ctx.familiarization
    )
    comm_prior = softmax_prior(0.86 * params["w_salience"] * comm_sal)

    # Utterance-specific individuated-anchor grounding matrix avail[u, r]:
    comp_w = (
        jnp.where(
            feat_ext == 2.0,
            1.0,
            jnp.where(feat_ext == 1.0, 0.75, jnp.where(feat_ext < n_obj, 0.15, 0.0)),
        )
        * real_mask
    )
    comp_fc = jnp.sum(real_lex * comp_w, axis=0, keepdims=True)
    fc_excess = jnp.maximum(comp_fc - 0.85, 0.0)
    base_avail = 1.0 / (1.0 + 0.50 * fc_excess)

    u_fc = jnp.where(real_lex > 0, fc[None, :], 100.0)
    min_u_fc = jnp.min(u_fc, axis=1, keepdims=True)
    is_min_u = real_lex * jnp.where(fc[None, :] == min_u_fc, 1.0, 0.0)
    n_min_u = jnp.maximum(jnp.sum(is_min_u, axis=1, keepdims=True), 1.0)
    min_u_cloned = jnp.clip(
        jnp.sum(is_min_u * (1.0 - is_singleton)[None, :], axis=1, keepdims=True),
        0.0,
        1.0,
    )

    u_overext = jnp.clip(feat_ext - 2.0, 0.0, 1.0) * real_mask
    u_univ = jnp.clip(feat_ext - 3.0, 0.0, 1.0) * real_mask
    rel_u_fc = jnp.clip(fc[None, :] - min_u_fc, 0.0, 2.0) / n_min_u

    uniq_boost = 1.0 + has_uniq[None, :] * (
        0.16 * u_overext * is_min_u
        + 0.04 * u_overext * min_u_cloned * (1.0 - is_min_u)
        + 0.32 * u_univ * min_u_cloned * (1.0 - is_min_u)
    )
    simp_pen = 1.0 + u_univ * (
        1.15 * rel_u_fc * (1.0 - 0.65 * min_u_cloned * has_uniq[None, :])
    )

    utt_avail = real_lex * base_avail * uniq_boost / simp_pen
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * utt_avail

    return real_mask, real_lex, n_obj, prior, comm_prior, avail


def listener(params, ctx):
    _, real_lex, n_obj, prior, comm_prior, avail = _scene_components(params, ctx)

    prag = L2(params["alpha"], avail, comm_prior)[ctx.utterance]
    lit0 = L0(avail)[ctx.utterance]

    heard_row = real_lex[ctx.utterance]
    heard_ext = jnp.sum(heard_row)
    overext = jnp.maximum(heard_ext - 2.0, 0.0)
    univ_crowd = jnp.maximum(heard_ext - 3.0, 0.0)
    is_4obj = jnp.where(n_obj >= 4, 1.0, 0.0)

    ret_l0 = is_4obj * (0.24 * jnp.clip(overext, 0.0, 1.0) + 0.26 * univ_crowd)
    ret_prior = is_4obj * (
        0.095 * jnp.where(heard_ext == 3.0, 1.0, 0.0) + 0.004 * univ_crowd
    )
    heard = (1.0 - ret_l0 - ret_prior) * prag + ret_l0 * lit0 + ret_prior * prior

    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    real_mask, _, _, _, comm_prior, avail = _scene_components(params, ctx)
    l1_tab = L1(params["alpha"], avail, comm_prior)
    w = real_mask * avail * jnp.exp(params["alpha"] * jnp.log(l1_tab + EPS))
    col_sum = jnp.sum(w, axis=0, keepdims=True)
    indifferent = real_mask / jnp.maximum(jnp.sum(real_mask), EPS)
    return jnp.where(col_sum > 0, w / jnp.maximum(col_sum, EPS), indifferent)
