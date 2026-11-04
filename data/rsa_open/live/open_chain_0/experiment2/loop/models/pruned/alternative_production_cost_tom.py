"""Pragmatic listener and speaker governed by competitive alternative production cost, prototype-preemption on overextended words, and figure-ground oddball salience."""

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
def L1[u: UTT, r: OBJ](alpha, avail: ..., prag_prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prag_prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(avail, u, r) * exp(alpha * log(L0[u, r](avail) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, avail: ..., prag_prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prag_prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(avail, u, r) * exp(alpha * log(L1[u, r](alpha, avail, prag_prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def _scene_components(params, ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask
    n_obj = ctx.lex.shape[1]
    n_real_utt = jnp.sum(real_mask[:, 0])
    denom = jnp.maximum(n_obj - 1.0, 1.0)
    fc = ctx.feature_count
    min_fc = jnp.min(fc)
    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    is_four = jnp.clip(n_obj - 3.0, 0.0, 1.0)

    # Pairwise perceptual dissimilarity and visual-type multiplicity.
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = (
        jnp.sum(diff, axis=0)
        + jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
        + jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    )
    clone_mult = jnp.sum(jnp.where(pair_dist == 0.0, 1.0, 0.0), axis=1)
    has_clones = jnp.where(jnp.max(clone_mult) > 1.0, 1.0, 0.0)
    is_pos_singleton = jnp.where((clone_mult == 1.0) & (fc > 0.0), 1.0, 0.0) * has_clones

    singleton_feat = jnp.where(feat_ext == 1.0, 1.0, 0.0) * real_mask
    sing_fc = jnp.sum(real_lex * singleton_feat, axis=0)
    has_singleton_feat = jnp.clip(sing_fc, 0.0, 1.0)
    is_disjoint_sing = jnp.where((fc == 1.0) & (sing_fc == 1.0), 1.0, 0.0)

    # Figure-ground perceptual contrast with polar negative-space oddball pop-out.
    mean_dist = jnp.sum(pair_dist, axis=1) / denom
    max_dist = jnp.max(pair_dist, axis=1)
    plain_gate = (
        1.5 / denom
        + 0.55 * jnp.maximum(mean_dist - 2.0, 0.0)
        + 0.26 * jnp.maximum(max_dist - 3.0, 0.0)
    )
    pos_gate = jnp.where(
        fc == 0,
        plain_gate,
        1.0 - 0.25 * jnp.exp(-fc),
    )
    contrast = (0.87 / jnp.sqrt(denom)) * pos_gate * jnp.sum(
        0.95 * (1.0 - jnp.exp(-pair_dist)) + 0.025 * pair_dist,
        axis=1,
    )
    oddity_pop = has_clones * (
        0.16 * has_singleton_feat
        + 0.06 * is_pos_singleton
    )

    has_univ = jnp.any((feat_ext == n_obj) * real_mask)
    var_mask = (feat_ext < n_obj) * real_mask
    var_fc = jnp.sum(real_lex * var_mask, axis=0)
    has_bg_anchor = jnp.where(min_fc == 1.0, 1.0, 0.22)

    # Majority-background exclusion oddball: in scenes lacking a universal feature
    # where a (n_obj - 1) majority feature defines the background group, a singleton-feature oddball pops out.
    maj_mask = jnp.where((n_obj >= 3) & (feat_ext == n_obj - 1.0), 1.0, 0.0) * real_mask
    has_single_maj = (1.0 - has_univ) * jnp.where(jnp.sum(maj_mask) == 1.0, 1.0, 0.0)
    lacks_maj = jnp.sum((1.0 - real_lex) * maj_mask, axis=0)
    maj_oddball = 0.25 * has_single_maj * lacks_maj * has_singleton_feat + 0.16 * is_four * has_clones * is_disjoint_sing

    base_salience = (
        contrast
        + oddity_pop
        + (0.22 + 0.58 * ctx.valence) * fc
        + 0.30 * (1.0 - ctx.grayscale)
        + 0.60 * ctx.familiarization
    )
    prag_prior = softmax_prior(params["w_salience"] * base_salience)

    prior_salience = (
        base_salience
        + 0.32 * has_univ * has_bg_anchor * jnp.maximum(var_fc - 2.0, 0.0)
        + maj_oddball
    )
    prior = softmax_prior(params["w_salience"] * prior_salience)

    # Speaker lexical retrieval: diagnostic features compete for selection while identical
    # visual clones share type-level lexical availability.
    diag_w = jnp.where(feat_ext <= 2.0, 1.0, jnp.where(feat_ext < n_obj, 0.11, 0.0)) * real_mask
    diag_fc = jnp.sum(real_lex * diag_w, axis=0, keepdims=True)
    fc_excess = jnp.maximum(diag_fc - 1.0, 0.0)
    type_share = 1.0 / (clone_mult[None, :] ** 0.11)
    true_avail = type_share / (1.0 + 0.50 * fc_excess)
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * (real_lex * true_avail)

    return (
        real_mask,
        real_lex,
        n_obj,
        n_real_utt,
        fc,
        clone_mult,
        has_clones,
        has_singleton_feat,
        diag_fc,
        prior,
        prag_prior,
        avail,
    )


def listener(params, ctx):
    (
        _,
        real_lex,
        n_obj,
        n_real_utt,
        fc,
        clone_mult,
        has_clones,
        has_singleton_feat,
        diag_fc,
        prior,
        prag_prior,
        avail,
    ) = _scene_components(params, ctx)

    l1_heard = L1(params["alpha"], avail, prag_prior)[ctx.utterance]
    l2_heard = L2(params["alpha"], avail, prag_prior)[ctx.utterance]

    heard_row = real_lex[ctx.utterance]
    heard_ext = jnp.sum(heard_row)
    overext = jnp.maximum(heard_ext - 2.0, 0.0)
    is_overext = jnp.clip(overext, 0.0, 1.0)
    univ_crowd = jnp.maximum(heard_ext - 3.0, 0.0)
    is_sub_univ = jnp.clip(n_obj - heard_ext, 0.0, 1.0) * is_overext
    is_four = jnp.clip(n_obj - 3.0, 0.0, 1.0)

    match_fc = jnp.where(heard_row > 0.0, fc, 100.0)
    min_match_fc = jnp.min(match_fc)
    max_match_fc = jnp.max(jnp.where(heard_row > 0.0, fc, -1.0))
    n_min_match = jnp.maximum(jnp.sum((heard_row > 0.0) & (fc == min_match_fc)), 1.0)
    has_complete_fig = jnp.where(max_match_fc == n_real_utt, 1.0, 0.0)
    has_polar_span = jnp.clip(has_complete_fig + jnp.where(max_match_fc == 2.0, 1.0, 0.0), 0.0, 1.0)
    unique_sole_proto = (
        is_four
        * (1.0 - has_clones)
        * has_polar_span
        * jnp.where((min_match_fc == 1.0) & (n_min_match == 1.0), 1.0, 0.0)
    )

    # When every matching candidate carries multiple features (min_match_fc > 1), counterfactual
    # simulation lacks a single-feature prototype anchor and partially truncates from L2 toward L1.
    no_single_feat_anchor = jnp.clip(min_match_fc - 1.0, 0.0, 1.0)
    trunc_l1 = 0.42 * is_overext * no_single_feat_anchor
    prag = (1.0 - trunc_l1) * l2_heard + trunc_l1 * l1_heard

    fc_gap = jnp.maximum(fc - min_match_fc, 0.0)
    rel_match_fc = jnp.minimum(fc_gap, 1.0 + 0.35 * jnp.maximum(fc_gap - 1.0, 0.0)) / n_min_match

    # Alternative-preemption grounding on overextended words: speakers reserve over-extended
    # words for minimal exemplars unless a unique singleton feature individuates a referent
    # against duplicate clones, while intermediate bundles between minimal and complete figure
    # are preempted by their narrower alternatives.
    is_unique_sing_scene = has_clones * jnp.where(jnp.sum(has_singleton_feat) == 1.0, 1.0, 0.0)
    shield_sing = 1.0 - 0.42 * univ_crowd * is_unique_sing_scene * has_singleton_feat
    is_intermediate = (
        has_complete_fig
        * jnp.where((heard_row > 0.0) & (fc > min_match_fc) & (fc < max_match_fc), 1.0, 0.0)
    )
    is_shared_nonclone = (
        is_unique_sing_scene
        * (1.0 - is_intermediate)
        * jnp.where((clone_mult == 1.0) & (has_singleton_feat == 0.0) & (fc > min_match_fc), 1.0, 0.0)
    )

    eff_diag_excess = jnp.maximum(
        diag_fc[0] - 0.80 - (0.45 + 0.50 * univ_crowd * is_unique_sing_scene) * has_clones * has_singleton_feat,
        0.0,
    )
    proto_excess = (0.18 * is_overext + 0.52 * univ_crowd + 0.16 * is_overext * unique_sole_proto) * (
        0.50 * eff_diag_excess
        + (1.85 + 1.40 * unique_sole_proto) * shield_sing * rel_match_fc
        + (1.75 - 0.85 * has_clones) * is_intermediate
        + 0.75 * univ_crowd * is_shared_nonclone
    )
    clone_div = clone_mult ** (0.16 * is_overext)
    oddball_anchor = 1.0 + (0.18 * is_overext + 0.16 * univ_crowd * is_unique_sing_scene) * has_clones * has_singleton_feat
    proto_w = heard_row * oddball_anchor / (clone_div * (1.0 + proto_excess))
    proto0 = proto_w / jnp.maximum(jnp.sum(proto_w), EPS)

    ret_proto = (0.16 + 0.16 * is_sub_univ + 0.10 * unique_sole_proto) * is_overext + 0.34 * univ_crowd
    ret_prior = 0.0028 * overext
    heard = (1.0 - ret_proto - ret_prior) * prag + ret_proto * proto0 + ret_prior * prag_prior

    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    real_mask, real_lex, _, _, _, _, _, _, _, _, prag_prior, avail = _scene_components(params, ctx)
    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    overext_u = jnp.clip(feat_ext - 2.0, 0.0, 1.0) * real_mask
    l1_tab = L1(params["alpha"], avail, prag_prior)
    l0_tab = avail / jnp.maximum(jnp.sum(avail, axis=1, keepdims=True), EPS)
    w_l0 = 0.14 + 0.22 * overext_u
    eff_l = (1.0 - w_l0) * l1_tab + w_l0 * l0_tab
    w = real_mask * avail * jnp.exp(params["alpha"] * jnp.log(eff_l + EPS))
    col_sum = jnp.sum(w, axis=0, keepdims=True)
    indifferent = real_mask / jnp.maximum(jnp.sum(real_mask), EPS)
    return jnp.where(col_sum > 0, w / jnp.maximum(col_sum, EPS), indifferent)
