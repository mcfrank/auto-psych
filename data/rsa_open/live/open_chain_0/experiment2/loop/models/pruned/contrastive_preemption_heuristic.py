"""Non-Bayesian contrastive-preemption and polar-prototype heuristic listener and speaker."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_salience": dist.LogNormal(0.0, 0.5),
}


@memo
def L_heur[u: UTT, r: OBJ](cue_fit: ..., rival_ratio: ..., prag_prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(cue_fit, u, r) * vec(prag_prior, r) / (1.0 + at(rival_ratio, u, r)),
    )
    return Pr[listener.r == r]


def _scene_components(params, ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask
    n_obj = ctx.lex.shape[1]
    n_real_utt = jnp.sum(real_mask[:, 0])
    denom = jnp.maximum(n_obj - 1.0, 1.0)
    fc = ctx.feature_count
    min_fc = jnp.min(fc)
    max_fc = jnp.max(fc)
    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)

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

    # Figure-ground perceptual contrast: on communicative trials, distance from a positive object
    # to a 0-feature blank ground foil merely duplicates feature count and is discounted.
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
    pair_kern = 0.95 * (1.0 - jnp.exp(-pair_dist)) + 0.025 * pair_dist
    is_blank_comp = jnp.where((fc[:, None] > 0) & (fc[None, :] == 0), 1.0, 0.0)
    comm_pair_w = 1.0 - 0.82 * is_blank_comp

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

    # Pre-utterance polar figure-ground prior salience: both the 0-feature negative-space pole
    # and the complete-figure pole attract attention on bipolar displays while intermediate
    # feature bundles sit in an attentional trough.
    is_four = jnp.clip(n_obj - 3.0, 0.0, 1.0)
    has_bg_anchor = has_univ * jnp.where(min_fc == 1.0, 1.0, 0.38 * (1.0 - has_clones))
    is_bipolar_scene = (
        (1.0 - has_clones)
        * jnp.where((min_fc == 0.0) & (max_fc == n_real_utt) & ((n_obj == 3) | (max_fc == 4.0)), 1.0, 0.0)
    )
    polar_extreme_boost = is_bipolar_scene * (
        (0.26 - 0.12 * is_four) * jnp.where(fc == 0.0, 1.0, 0.0)
        + 0.18 * is_four * jnp.where(fc == 4.0, 1.0, 0.0)
        - (0.09 - 0.05 * is_four) * jnp.where((fc > 0.0) & (fc < max_fc), 1.0, 0.0)
    )
    is_three_clone_sing = jnp.where(n_obj == 3, 1.0, 0.0) * is_pos_singleton * has_singleton_feat
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

    # Feature-to-object availability: diagnostic cue dilution over visual types with
    # blank-ground contrast and intermediate-bundle suppression on over-extended features.
    diag_w = jnp.where(feat_ext <= 2.0, 1.0, jnp.where(feat_ext < n_obj, 0.11, 0.0)) * real_mask
    diag_fc = jnp.sum(real_lex * diag_w, axis=0, keepdims=True)
    has_plain = jnp.where(min_fc == 0.0, 1.0, 0.0)
    fc_excess = jnp.maximum(diag_fc - 1.0, 0.0) * (
        1.0 + 0.30 * has_plain * jnp.where(feat_ext == 2.0, 1.0, 0.0)
    )

    min_fc_u = jnp.min(jnp.where(real_lex > 0.0, fc[None, :], 100.0), axis=1, keepdims=True)
    max_fc_u = jnp.max(jnp.where(real_lex > 0.0, fc[None, :], -1.0), axis=1, keepdims=True)
    has_full_fig_u = jnp.where(max_fc_u == n_real_utt, 1.0, 0.0)
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

    # Non-Bayesian two-stage contrastive preemption heuristic:
    # Stage 1: Each feature divides its cue validity across its bearers in proportion to avail.
    alpha = params["alpha"]
    share0 = avail / jnp.maximum(jnp.sum(avail, axis=1, keepdims=True), EPS)

    # Stage 2: Each object claims features in proportion to their first-order cue validity;
    # unclaimed cue association across objects then drives sublinear rival preemption.
    claim_w = avail * jnp.exp(alpha * jnp.log(share0 + EPS))
    claim = claim_w / jnp.maximum(jnp.sum(claim_w, axis=0, keepdims=True), EPS)
    assoc_w = claim * prag_prior[None, :]
    assoc = assoc_w / jnp.maximum(jnp.sum(assoc_w, axis=1, keepdims=True), EPS)

    own_strength = avail * jnp.exp(alpha * jnp.log(assoc + EPS))
    total_strength = jnp.sum(own_strength, axis=0, keepdims=True)
    rival_strength = jnp.maximum(total_strength - own_strength, 0.0)
    n_rivals = jnp.maximum(fc[None, :] - 1.0, 1.0)
    eff_rival = rival_strength / (n_rivals ** (0.08 * (1.0 - has_plain)))
    safe_own = jnp.where(ctx.lex > 0.0, jnp.maximum(own_strength, EPS), 1.0)
    rival_ratio = ctx.lex * (eff_rival / safe_own)
    cue_fit = ctx.lex

    return (
        real_mask,
        real_lex,
        n_obj,
        n_real_utt,
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
        cue_fit,
        rival_ratio,
        assoc,
    )


def listener(params, ctx):
    (
        _,
        real_lex,
        n_obj,
        n_real_utt,
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
        _,
        cue_fit,
        rival_ratio,
        _,
    ) = _scene_components(params, ctx)

    heur = L_heur(cue_fit, rival_ratio, prag_prior)[ctx.utterance]

    # Prototype-simplicity category grounding on over-extended utterances (heard_ext >= 3).
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

    # Same-tier unanchored non-clone distractor shadowing:
    shadowed_same_tier = jnp.where(
        (heard_row > 0.0)
        & (fc > min_match_fc)
        & (fc < n_real_utt)
        & (clone_mult == 1.0)
        & (has_singleton_feat == 0.0)
        & (same_match_fc >= 2.0)
        & ((has_clones > 0.0) | (any_sing_feat > 0.0)),
        1.0,
        0.0,
    )
    shadowed_min_clone = (
        univ_crowd
        * is_pos_singleton
        * (1.0 - has_singleton_feat)
        * any_sing_feat
        * jnp.where(fc == min_match_fc, 0.35, 0.0)
    )
    complete_sing_fig = has_clones * has_singleton_feat * jnp.where(fc == n_real_utt, 1.0, 0.0)

    eff_diag_excess = jnp.maximum(diag_fc[0] - 0.80 - 0.45 * has_clones * has_singleton_feat, 0.0)
    overext_scale = (
        (
            0.16
            + 0.28 * unique_sole_proto
            + 0.24 * is_sub_univ * is_scene_sole_proto
            + 0.25 * nonmin_sole_proto
            + 0.75 * multi_clone_min
        )
        * is_overext
        + 0.50 * univ_crowd
    )
    rel_shield = 1.0 - 0.55 * is_sub_univ * sole_clone_min * has_singleton_feat - 0.38 * is_uniq_tier_after_clone_min
    lit_excess = overext_scale * (
        0.50 * eff_diag_excess
        + 1.85 * rel_shield * rel_match_fc
        + 1.15 * is_bipolar_mid
        + 3.80 * shadowed_same_tier
        + 1.10 * shadowed_min_clone
    )
    clone_div = clone_mult ** (0.16 * is_overext * (1.0 - multi_clone_min))
    oddball_anchor = (
        1.0
        + 0.10 * is_uniq_tier_after_clone_min
        + is_overext * (1.0 - multi_clone_min) * has_clones * has_singleton_feat * (
            0.18 + 0.65 * is_sub_univ * sole_clone_min + 0.95 * univ_crowd * complete_sing_fig
        )
    )
    true_lit_w = heard_row * oddball_anchor / (clone_div * (1.0 + lit_excess))

    # Prototype-scanning spillover onto simple excluded foils on 4-object sub-universal words:
    # when the excluded foil is strictly simpler than all matching candidates (fc < min_match_fc),
    # prototype-anchored grounding retains spillover onto that sole minimal prototype foil.
    is_four_sub = is_sub_univ * jnp.where(n_obj == 4, 1.0, 0.0)
    is_sub_min_foil = jnp.where(fc < min_match_fc, 1.0, 0.0)
    false_lit_w = 0.19 * is_four_sub * is_sub_min_foil * (1.0 - heard_row) * jnp.sum(true_lit_w)
    lit_w = true_lit_w + false_lit_w
    lit0 = lit_w / jnp.maximum(jnp.sum(lit_w), EPS)

    ret_l0 = (0.16 + 0.16 * is_sub_univ + 0.10 * is_sub_univ * is_scene_sole_proto) * is_overext + 0.34 * univ_crowd
    ret_prior = 0.0028 * overext
    heard = (1.0 - ret_l0 - ret_prior) * heur + ret_l0 * lit0 + ret_prior * prag_prior

    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    real_mask, _, _, _, _, _, _, _, _, _, _, _, _, _, avail, _, rival_ratio, assoc = _scene_components(params, ctx)
    l0_tab = avail / jnp.maximum(jnp.sum(avail, axis=1, keepdims=True), EPS)
    eff_l = 0.72 * assoc + 0.28 * l0_tab
    w = real_mask * avail * jnp.exp(params["alpha"] * jnp.log(eff_l + EPS)) / (1.0 + 0.14 * rival_ratio)
    col_sum = jnp.sum(w, axis=0, keepdims=True)
    indifferent = real_mask / jnp.maximum(jnp.sum(real_mask), EPS)
    return jnp.where(col_sum > 0, w / jnp.maximum(col_sum, EPS), indifferent)
