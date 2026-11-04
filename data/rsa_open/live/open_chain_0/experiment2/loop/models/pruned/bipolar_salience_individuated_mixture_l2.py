"""Heterogeneous listener and speaker population with bipolar figure-ground salience, prototype-preempted category grounding, and sub-universal foil fallback."""

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
    is_full_fig = jnp.where(fc == n_real_utt, 1.0, 0.0)
    complete_sing_fig = has_clones * has_singleton_feat * is_full_fig

    # Figure-ground perceptual contrast with polar negative-space and disjoint-singleton oddball pop-out.
    mean_dist = jnp.sum(pair_dist, axis=1) / denom
    max_dist = jnp.max(pair_dist, axis=1)
    plain_gate = (
        1.5 / denom
        + 0.55 * jnp.maximum(mean_dist - 2.0, 0.0)
        + 0.22 * jnp.maximum(max_dist - 3.0, 0.0)
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
        0.12 * has_singleton_feat
        + 0.06 * is_pos_singleton
        + 0.30 * is_disjoint_sing
        + 0.14 * complete_sing_fig
    )

    base_salience = (
        contrast
        + oddity_pop
        + (0.22 + 0.58 * ctx.valence) * fc
        + 0.30 * (1.0 - ctx.grayscale)
        + 0.60 * ctx.familiarization
    )
    prag_prior = softmax_prior(params["w_salience"] * base_salience)

    # Bipolar figure-ground organization on uninformative prior displays: polar extremes
    # (complete figures and zero-feature ground) stand out against intermediate-complexity bundles.
    has_prior_mid = (
        is_four
        * (1.0 - has_clones)
        * jnp.where(max_fc == n_real_utt, 1.0, 0.0)
        * jnp.any((fc > min_fc) & (fc < n_real_utt))
    )
    is_prior_mid = jnp.where((fc > 1.0) & (fc < n_real_utt), 1.0, 0.0)
    prior_salience = (
        base_salience
        + 0.24 * has_prior_mid * is_full_fig
        - 0.12 * has_prior_mid * jnp.where(n_real_utt == 4.0, 1.0, 0.0) * is_prior_mid
    )
    prior = softmax_prior(params["w_salience"] * prior_salience)

    # Pragmatic speaker lexical retrieval: diagnostic cue competition over visual types
    # with sole-prototype preemption of unindividuated shared-feature distractors on over-extended utterances.
    diag_w = jnp.where(feat_ext <= 2.0, 1.0, jnp.where(feat_ext < n_obj, 0.11, 0.0)) * real_mask
    diag_fc = jnp.sum(real_lex * diag_w, axis=0, keepdims=True)
    fc_excess = jnp.maximum(diag_fc - 1.0, 0.0)

    min_fc_u = jnp.min(jnp.where(real_lex > 0.0, fc[None, :], 100.0), axis=1, keepdims=True)
    max_fc_u = jnp.max(jnp.where(real_lex > 0.0, fc[None, :], -1.0), axis=1, keepdims=True)
    n_min_u = jnp.sum(jnp.where((real_lex > 0.0) & (fc[None, :] == min_fc_u), 1.0, 0.0), axis=1, keepdims=True)
    has_full_fig_u = jnp.where(max_fc_u == n_real_utt, 1.0, 0.0)
    is_shared_mid_u = (
        is_four
        * jnp.where((real_lex > 0.0) & (fc[None, :] == min_fc_u + 1.0) & (n_min_u == 1.0), 1.0, 0.0)
        * (1.0 - has_full_fig_u)
        * jnp.where(clone_mult[None, :] == 1.0, 1.0, 0.0)
        * (1.0 - has_singleton_feat[None, :])
    )
    overext_u = jnp.clip(feat_ext - 2.0, 0.0, 1.0) * real_mask
    mid_pen_u = 0.30 * overext_u * is_shared_mid_u

    type_share = 1.0 / (clone_mult[None, :] ** (0.12 * is_four))
    true_avail = type_share / ((1.0 + 0.50 * fc_excess) * (1.0 + mid_pen_u))
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * (real_lex * true_avail)

    return (
        real_mask,
        real_lex,
        n_obj,
        n_real_utt,
        fc,
        min_fc,
        is_four,
        clone_mult,
        has_clones,
        has_singleton_feat,
        any_sing_feat,
        is_pos_singleton,
        complete_sing_fig,
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
        min_fc,
        is_four,
        clone_mult,
        has_clones,
        has_singleton_feat,
        any_sing_feat,
        is_pos_singleton,
        complete_sing_fig,
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
    fc_gap = jnp.maximum(fc - min_match_fc, 0.0)
    rel_match_fc = jnp.minimum(fc_gap, 1.0 + 0.35 * jnp.maximum(fc_gap - 1.0, 0.0)) / n_min_match

    has_full_fig = jnp.where(max_match_fc == n_real_utt, 1.0, 0.0)
    is_univ_mid = (
        univ_crowd
        * has_full_fig
        * jnp.where((heard_row > 0.0) & (fc > min_match_fc) & (fc < max_match_fc), 1.0, 0.0)
    )
    is_shared_mid = (
        is_four
        * jnp.where((heard_row > 0.0) & (fc == min_match_fc + 1.0) & (n_min_match == 1.0), 1.0, 0.0)
        * (1.0 - has_full_fig)
        * jnp.where(clone_mult == 1.0, 1.0, 0.0)
        * (1.0 - has_singleton_feat)
    )
    unique_sole_proto = is_sub_univ * jnp.where(
        (n_obj == 4) & (min_match_fc == 1.0) & (n_min_match == 1.0) & (has_clones == 0.0) & (max_match_fc != 3.0),
        1.0,
        0.0,
    )
    multi_clone_min = is_sub_univ * jnp.where((min_match_fc > min_fc) & (n_min_match > 1.0), 1.0, 0.0)
    sole_clone_min = is_overext * jnp.where((min_match_fc == 1.0) & (n_min_match > 1.0), 1.0, 0.0)
    shadowed_nonclone = (
        is_pos_singleton
        * (1.0 - has_singleton_feat)
        * jnp.where(fc > min_match_fc, 1.0 - has_full_fig, 0.40 * any_sing_feat)
    )

    eff_diag_excess = jnp.maximum(diag_fc[0] - 0.80 - 0.45 * has_clones * has_singleton_feat, 0.0)
    overext_scale = (0.16 + 0.28 * unique_sole_proto + 0.55 * multi_clone_min) * is_overext + 0.50 * univ_crowd
    lit_excess = (1.0 - 0.65 * is_univ_mid) * overext_scale * (
        0.50 * eff_diag_excess
        + 1.85 * (1.0 - 0.55 * sole_clone_min * has_singleton_feat) * rel_match_fc
        + 1.15 * is_shared_mid
        + 1.10 * univ_crowd * shadowed_nonclone
    )
    clone_exp = (0.16 + 0.36 * jnp.clip(fc - min_match_fc, 0.0, 1.0)) * is_four * is_overext * (1.0 - multi_clone_min)
    clone_div = clone_mult ** clone_exp
    oddball_anchor = 1.0 + is_four * is_overext * (1.0 - multi_clone_min) * has_clones * has_singleton_feat * (
        0.18 + 0.55 * is_sub_univ * sole_clone_min + 0.95 * univ_crowd * complete_sing_fig
    )
    lit_w = heard_row * oddball_anchor / (clone_div * (1.0 + lit_excess))
    lit0 = lit_w / jnp.maximum(jnp.sum(lit_w), EPS)

    # Population mixture over pragmatic (L2), literal (L0), and topic-anchored (prior) listener types,
    # with salience-weighted foil intrusion under three-referent over-extension in four-object scenes.
    foil_prior = (1.0 - heard_row) * prag_prior
    foil_dist = foil_prior / jnp.maximum(jnp.sum(foil_prior), EPS)
    ret_l0 = (0.16 + 0.16 * is_sub_univ) * is_overext + 0.34 * univ_crowd
    ret_foil = 0.016 * is_sub_univ
    ret_prior = 0.0028 * overext
    heard = (
        (1.0 - ret_l0 - ret_prior - ret_foil) * prag
        + ret_l0 * lit0
        + ret_prior * prag_prior
        + ret_foil * foil_dist
    )

    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    real_mask, _, _, _, _, _, _, _, _, _, _, _, _, _, _, prag_prior, avail = _scene_components(params, ctx)
    l1_tab = L1(params["alpha"], avail, prag_prior)
    l0_tab = avail / jnp.maximum(jnp.sum(avail, axis=1, keepdims=True), EPS)
    eff_l = 0.78 * l1_tab + 0.22 * l0_tab
    w = real_mask * avail * jnp.exp(params["alpha"] * jnp.log(eff_l + EPS))
    col_sum = jnp.sum(w, axis=0, keepdims=True)
    indifferent = real_mask / jnp.maximum(jnp.sum(real_mask), EPS)
    return jnp.where(col_sum > 0, w / jnp.maximum(col_sum, EPS), indifferent)
