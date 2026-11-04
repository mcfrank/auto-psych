"""Polar figure-ground topic prior Theory of Mind with bipolar negative-space and complete-figure anchoring, sub-maximal prototype contrast, second-tier distractor shadowing, and unanchored sub-universal boundary spillover."""

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
    n_real_utt = jnp.sum(real_mask)
    denom = jnp.maximum(n_obj - 1.0, 1.0)
    fc = ctx.feature_count
    min_fc = jnp.min(fc)
    max_fc = jnp.max(fc)
    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    is_four = jnp.clip(n_obj - 3.0, 0.0, 1.0)
    is_three = jnp.where(n_obj == 3, 1.0, 0.0)

    # Pairwise perceptual dissimilarity and visual-type multiplicity across identical clones.
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
    any_sing_feat = jnp.max(has_singleton_feat)
    is_disjoint_sing = jnp.where((fc == 1.0) & (sing_fc == 1.0), 1.0, 0.0)

    # Figure-ground perceptual contrast: negative framing elevates 0-feature ground oddballs,
    # whereas on communicative trials distance from a positive object to a 0-feature blank foil is discounted.
    mean_dist = jnp.sum(pair_dist, axis=1) / denom
    max_dist = jnp.max(pair_dist, axis=1)
    neg_val = jnp.maximum(-ctx.valence, 0.0)
    plain_gate = (
        1.5 / denom
        + 0.55 * jnp.maximum(mean_dist - 2.0, 0.0)
        + 0.26 * jnp.maximum(max_dist - 3.0, 0.0)
        + 0.75 * neg_val
    )
    pos_gate = jnp.where(
        fc == 0,
        plain_gate,
        1.0 - 0.25 * jnp.exp(-fc),
    )
    pair_kern = 0.95 * (1.0 - jnp.exp(-pair_dist)) + 0.025 * pair_dist
    is_blank_comp = jnp.where((fc[:, None] > 0) & (fc[None, :] == 0), 1.0, 0.0)
    comm_pair_w = 1.0 - 0.80 * is_blank_comp

    prior_contrast = (0.87 / jnp.sqrt(denom)) * pos_gate * jnp.sum(pair_kern, axis=1)
    comm_contrast = (0.87 / jnp.sqrt(denom)) * pos_gate * jnp.sum(comm_pair_w * pair_kern, axis=1)

    oddity_pop = has_clones * (
        0.12 * has_singleton_feat
        + 0.06 * is_pos_singleton
        + 0.30 * is_disjoint_sing
    )

    has_univ = jnp.any((feat_ext == n_obj) * real_mask)
    var_mask = (feat_ext < n_obj) * real_mask
    var_fc = jnp.sum(real_lex * var_mask, axis=0)

    base_salience = (
        comm_contrast
        + oddity_pop
        + (0.22 + 0.58 * ctx.valence) * fc
        + 0.30 * (1.0 - ctx.grayscale)
        + 0.60 * ctx.familiarization
    )
    prag_prior = softmax_prior(params["w_salience"] * base_salience)

    # Pre-utterance polar figure-ground prior salience across 3- and 4-object displays:
    n_scene_min = jnp.sum(fc == min_fc)
    has_bg_anchor = has_univ * jnp.where(
        min_fc == 1.0,
        jnp.where(n_scene_min <= 2.0, 1.60, 1.12),
        0.28 * (1.0 - has_clones),
    )
    is_bipolar_scene = is_four * (1.0 - has_clones) * jnp.where((min_fc == 0.0) & (max_fc == n_real_utt), 1.0, 0.0)
    is_bipolar_three = is_three * (1.0 - has_clones) * jnp.where((min_fc == 0.0) & (max_fc == n_real_utt), 1.0, 0.0)
    is_elevated_polar_scene = (
        is_four * (1.0 - has_clones) * jnp.where((min_fc == 2.0) & (max_fc == 4.0) & (n_real_utt == 4.0), 1.0, 0.0)
    )
    is_three_sole_min_scene = (
        is_three * (1.0 - has_clones) * jnp.where((min_fc == 1.0) & (max_fc == 2.0) & (n_real_utt == 3.0), 1.0, 0.0)
    )
    subfour_blank_pen = -0.35 * is_four * jnp.where((fc == 0.0) & (max_fc < 4.0) & (n_scene_min == 1.0), 1.0, 0.0)
    polar_extreme_boost = (
        is_bipolar_scene * (
            0.24 * jnp.where((fc == 0.0) & (max_fc == 4.0), 1.0, 0.0)
            - 0.11 * jnp.where((fc > 0.0) & (fc < max_fc) & (max_fc == 4.0), 1.0, 0.0)
            + jnp.where(max_fc == 4.0, 0.14, 0.22) * jnp.where(fc == n_real_utt, 1.0, 0.0)
        )
        + subfour_blank_pen
        + is_bipolar_three * (
            0.22 * jnp.where(fc == 0.0, 1.0, 0.0)
            - 0.08 * jnp.where((fc > 0.0) & (fc < max_fc), 1.0, 0.0)
        )
        + is_elevated_polar_scene * (
            0.08 * jnp.where(fc == min_fc, 1.0, 0.0)
            - 0.16 * jnp.where((fc > min_fc) & (fc < max_fc), 1.0, 0.0)
        )
        + is_three_sole_min_scene * (
            0.34 * jnp.where(fc == 1.0, 1.0, 0.0)
            - 0.18 * jnp.where((fc == 2.0) & (has_singleton_feat == 0.0), 1.0, 0.0)
        )
    )
    is_three_clone_sing = is_three * is_pos_singleton * has_singleton_feat
    prior_salience = (
        prior_contrast
        + oddity_pop
        + 0.25 * is_three_clone_sing
        + (0.22 + 0.58 * ctx.valence) * fc
        + 0.32 * has_bg_anchor * jnp.maximum(var_fc - 2.0, 0.0)
        + polar_extreme_boost
        + 0.30 * (1.0 - ctx.grayscale)
        + 0.60 * ctx.familiarization
    )
    prior = softmax_prior(params["w_salience"] * prior_salience)

    # Pragmatic speaker lexical retrieval: diagnostic cue competition over visual types
    # with blank-anchor two-referent contrast, unique scene-maximal singleton figure shielding,
    # unanchored sub-maximal competitor dilution, and bipolar intermediate-bundle suppression.
    diag_w = jnp.where(feat_ext <= 2.0, 1.0, jnp.where(feat_ext < n_obj, 0.11, 0.0)) * real_mask
    diag_fc = jnp.sum(real_lex * diag_w, axis=0, keepdims=True)
    has_plain = jnp.where(min_fc == 0.0, 1.0, 0.0)
    has_uniq_max = jnp.where(jnp.sum(fc == max_fc) == 1.0, 1.0, 0.0)

    min_fc_u = jnp.min(jnp.where(real_lex > 0.0, fc[None, :], 100.0), axis=1, keepdims=True)
    max_fc_u = jnp.max(jnp.where(real_lex > 0.0, fc[None, :], -1.0), axis=1, keepdims=True)
    has_full_fig_u = jnp.where(max_fc_u == n_real_utt, 1.0, 0.0)

    submax_sing_fig_u = (
        is_four
        * (1.0 - has_clones)
        * any_sing_feat
        * has_uniq_max
        * has_singleton_feat[None, :]
        * jnp.where((feat_ext == 2.0) & (max_fc < n_real_utt) & (fc[None, :] == max_fc), 1.0, 0.0)
    )
    unanchored_submax_comp_u = (
        is_four
        * (1.0 - has_clones)
        * jnp.where(
            (feat_ext == 2.0)
            & (min_fc_u == 1.0)
            & (max_fc < n_real_utt)
            & (fc[None, :] == max_fc_u)
            & (has_singleton_feat[None, :] == 0.0),
            jnp.where(max_fc_u < max_fc, -0.15, 0.12),
            0.0,
        )
    )
    fc_excess = jnp.maximum(
        diag_fc - 1.0 - 0.72 * submax_sing_fig_u + unanchored_submax_comp_u,
        0.0,
    ) * (1.0 + 0.28 * has_plain * jnp.where(feat_ext == 2.0, 1.0, 0.0))

    is_bipolar_mid_u = (
        jnp.where((real_lex > 0.0) & (fc[None, :] > min_fc_u) & (fc[None, :] < max_fc_u), 1.0, 0.0)
        * has_full_fig_u
        * (1.0 - has_clones)
    )
    overext_u = jnp.clip(feat_ext - 2.0, 0.0, 1.0) * real_mask
    mid_pen_u = 0.25 * overext_u * is_bipolar_mid_u

    type_share = 1.0 / (clone_mult[None, :] ** 0.12)
    true_avail = type_share / ((1.0 + 0.50 * fc_excess) * (1.0 + mid_pen_u))
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * (real_lex * true_avail)

    return (
        real_mask,
        real_lex,
        n_obj,
        n_real_utt,
        is_four,
        overext_u,
        fc,
        min_fc,
        clone_mult,
        has_clones,
        has_singleton_feat,
        any_sing_feat,
        is_pos_singleton,
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
        is_four,
        _,
        fc,
        min_fc,
        clone_mult,
        has_clones,
        has_singleton_feat,
        any_sing_feat,
        is_pos_singleton,
        diag_fc,
        prior,
        prag_prior,
        avail,
    ) = _scene_components(params, ctx)

    prag = L2(params["alpha"], avail, prag_prior)[ctx.utterance]

    # Shallow literal listener type: prototype-graded category grounding on over-extended utterances.
    heard_row = real_lex[ctx.utterance]
    heard_ext = jnp.sum(heard_row)
    overext = jnp.maximum(heard_ext - 2.0, 0.0)
    is_overext = jnp.clip(overext, 0.0, 1.0)
    univ_crowd = jnp.maximum(heard_ext - 3.0, 0.0)
    is_sub_univ = jnp.clip(n_obj - heard_ext, 0.0, 1.0) * is_overext

    match_fc = jnp.where(heard_row > 0.0, fc, 100.0)
    min_match_fc = jnp.min(match_fc)
    max_match_fc = jnp.max(jnp.where(heard_row > 0.0, fc, -1.0))
    n_min_match = jnp.maximum(jnp.sum((heard_row > 0.0) & (fc == min_match_fc)), 1.0)
    n_scene_min = jnp.sum(fc == min_fc)
    fc_gap = jnp.maximum(fc - min_match_fc, 0.0)
    rel_match_fc = jnp.minimum(fc_gap, 1.0 + 0.35 * jnp.maximum(fc_gap - 1.0, 0.0)) / n_min_match

    same_match_fc = jnp.sum(
        jnp.where((heard_row[:, None] > 0.0) & (fc[:, None] == fc[None, :]), 1.0, 0.0),
        axis=0,
    )

    has_full_fig = jnp.where(max_match_fc == n_real_utt, 1.0, 0.0)
    is_bipolar_mid = (
        jnp.where((heard_row > 0.0) & (fc > min_match_fc) & (fc < max_match_fc), 1.0, 0.0)
        * has_full_fig
        * (1.0 - has_clones)
    )
    unique_sole_proto = jnp.where(
        (n_obj == 4) & (min_match_fc == 1.0) & (n_min_match == 1.0) & (has_clones == 0.0) & (max_match_fc != 3.0),
        1.0,
        0.0,
    )
    is_scene_sole_proto = unique_sole_proto * jnp.where(n_scene_min == 1.0, 1.0, 0.0)
    nonmin_sole_proto = is_sub_univ * jnp.where((min_match_fc > min_fc) & (n_min_match == 1.0), 1.0, 0.0)
    multi_clone_min = is_sub_univ * jnp.where((min_match_fc > min_fc) & (n_min_match > 1.0), 1.0, 0.0)
    sole_clone_min = is_overext * jnp.where((min_match_fc == 1.0) & (n_min_match > 1.0), 1.0, 0.0)
    is_uniq_tier_after_clone_min = (
        univ_crowd * sole_clone_min * jnp.where((fc == 2.0) & (same_match_fc == 1.0), 1.0, 0.0)
    )
    peer_tier_after_clone_min = (
        univ_crowd * sole_clone_min * jnp.where((fc == 2.0) & (same_match_fc >= 2.0), 1.0, 0.0)
    )

    # Same-tier unanchored non-clone distractor shadowing and singleton-peer elevation:
    shadowed_same_tier = jnp.where(
        (heard_row > 0.0)
        & (fc > min_match_fc)
        & (fc < n_real_utt)
        & (clone_mult == 1.0)
        & (has_singleton_feat == 0.0)
        & ((same_match_fc >= 2.0) | ((fc < max_match_fc) & (is_bipolar_mid == 0.0) & (has_clones == 0.0)))
        & ((has_clones > 0.0) | (any_sing_feat > 0.0)),
        1.0,
        0.0,
    )
    has_unanchored_peer = jnp.max(shadowed_same_tier)
    sing_peer_anchor = (
        (1.0 - has_clones)
        * has_unanchored_peer
        * has_singleton_feat
        * jnp.where((heard_row > 0.0) & (same_match_fc >= 2.0), 1.0, 0.0)
    )
    shadowed_min_clone = (
        univ_crowd
        * is_pos_singleton
        * (1.0 - has_singleton_feat)
        * any_sing_feat
        * jnp.where(fc == min_match_fc, 0.35, 0.0)
    )
    shadowed_second_tier_nonclone = (
        univ_crowd
        * has_clones
        * (1.0 - has_singleton_feat)
        * jnp.where((clone_mult == 1.0) & (fc == min_match_fc + 1.0) & (min_match_fc >= 2.0), 0.65, 0.0)
    )
    complete_sing_fig = has_clones * has_singleton_feat * jnp.where(fc == n_real_utt, 1.0, 0.0)

    eff_diag_excess = jnp.maximum(
        diag_fc[0] - 0.80 - 0.45 * (has_clones * has_singleton_feat + sing_peer_anchor),
        0.0,
    )
    overext_scale = (
        (
            0.16
            + 0.28 * unique_sole_proto
            + 0.58 * is_sub_univ * is_scene_sole_proto
            + 0.25 * nonmin_sole_proto
            + 1.15 * multi_clone_min
        )
        * is_overext
        + 0.50 * univ_crowd
    )
    rel_shield = (
        1.0
        - 0.55 * is_sub_univ * sole_clone_min * has_singleton_feat
        - 0.35 * is_uniq_tier_after_clone_min
        - 0.35 * sing_peer_anchor
    )
    lit_excess = overext_scale * (
        0.50 * eff_diag_excess
        + 1.85 * rel_shield * rel_match_fc
        + 1.15 * is_bipolar_mid
        + 3.20 * shadowed_same_tier
        + 1.10 * shadowed_min_clone
        + 0.55 * peer_tier_after_clone_min
        + 0.65 * shadowed_second_tier_nonclone
    )
    clone_div = clone_mult ** (0.16 * is_overext * (1.0 - multi_clone_min))
    oddball_anchor = (
        1.0
        + 0.08 * is_uniq_tier_after_clone_min
        + 0.35 * sing_peer_anchor
        + is_overext * (1.0 - multi_clone_min) * has_clones * has_singleton_feat * (
            0.18 + 0.55 * is_sub_univ * sole_clone_min + 0.95 * univ_crowd * complete_sing_fig
        )
    )
    true_lit_w = heard_row * oddball_anchor / (clone_div * (1.0 + lit_excess))

    # Sub-universal prototype-boundary spillover: when a 3-of-4 word excludes a simpler
    # prototype (fc < min_match_fc) or a simple/intermediate foil (fc <= 2) in a clone-free display,
    # shallow literal grounding retains soft weight on the excluded foil.
    sub_univ_foil = (
        is_four
        * is_sub_univ
        * (1.0 - heard_row)
        * jnp.where(
            fc < min_match_fc,
            0.19,
            jnp.where((has_clones == 0.0) & (fc <= 2.0), 0.085, 0.0),
        )
    )
    false_lit_w = sub_univ_foil * jnp.sum(true_lit_w)
    lit_w = true_lit_w + false_lit_w
    lit0 = lit_w / jnp.maximum(jnp.sum(lit_w), EPS)

    # Population mixture over pragmatic (L2), literal (L0), and topic-anchored (prior) listener types.
    ret_l0 = (0.16 + 0.16 * is_sub_univ + 0.16 * is_sub_univ * is_scene_sole_proto) * is_overext + 0.34 * univ_crowd
    ret_prior = 0.0018 * overext
    heard = (1.0 - ret_l0 - ret_prior) * prag + ret_l0 * lit0 + ret_prior * prag_prior

    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    real_mask, _, _, _, _, overext_u, _, _, _, _, _, _, _, _, _, prag_prior, avail = _scene_components(params, ctx)
    l1_tab = L1(params["alpha"], avail, prag_prior)
    l0_tab = avail / jnp.maximum(jnp.sum(avail, axis=1, keepdims=True), EPS)
    w_l0 = 0.20 + 0.14 * overext_u
    eff_l = (1.0 - w_l0) * l1_tab + w_l0 * l0_tab
    w = real_mask * avail * jnp.exp(params["alpha"] * jnp.log(eff_l + EPS))
    col_sum = jnp.sum(w, axis=0, keepdims=True)
    indifferent = real_mask / jnp.maximum(jnp.sum(real_mask), EPS)
    return jnp.where(col_sum > 0, w / jnp.maximum(col_sum, EPS), indifferent)
