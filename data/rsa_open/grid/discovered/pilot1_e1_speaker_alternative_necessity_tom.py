"""Pragmatic listener and speaker governed by alternative-word competition, sole-label universal naming necessity, and contrastive figure salience."""

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

    # Contrastive figure salience: universal scene-wide features form a shared background frame,
    # enhancing the figure salience of objects that possess all contrastive (varying) features.
    var_mask = jnp.where((feat_ext > 0.0) & (feat_ext < n_obj), 1.0, 0.0) * real_mask
    var_fc = jnp.sum(real_lex * var_mask, axis=0)
    has_univ = jnp.max(jnp.where(feat_ext == n_obj, 1.0, 0.0) * real_mask)

    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = (
        jnp.sum(diff, axis=0)
        + jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
        + jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    )
    mean_dist = jnp.sum(pair_dist, axis=1) / denom
    plain_gate = 1.5 / denom + 0.55 * jnp.maximum(mean_dist - 2.0, 0.0)
    pos_gate = jnp.where(fc == 0, plain_gate, 1.0 - 0.25 * jnp.exp(-fc))
    contrast = (0.88 / jnp.sqrt(denom)) * pos_gate * jnp.sum(1.0 - jnp.exp(-pair_dist), axis=1)

    fig_proto = has_univ * jnp.maximum(var_fc - 2.0, 0.0)
    salience = (
        contrast
        + (0.22 + 0.50 * ctx.valence) * fc
        + 0.36 * fig_proto
        + 0.30 * (1.0 - ctx.grayscale)
        + 0.58 * ctx.familiarization
    )
    prior = softmax_prior(params["w_salience"] * salience)

    # Speaker diagnostic alternative competition: narrow features (ext <= 2) compete strongly (1.0),
    # partially diagnostic 3-object features compete weakly (0.10), universal features do not (0.0).
    diag_w = jnp.where(feat_ext <= 2.0, 1.0, jnp.where(feat_ext < n_obj, 0.10, 0.0)) * real_mask
    diag_fc = jnp.sum(real_lex * diag_w, axis=0, keepdims=True)
    fc_excess = jnp.maximum(diag_fc - 1.0, 0.0)
    true_avail = 1.0 / (1.0 + 0.50 * fc_excess)
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * (real_lex * true_avail)

    # Sole-label universal naming necessity: when a scene contains a unique 1-feature referent (n_sole == 1)
    # whose only available word is the universal label, any referent with another available word (fc > 1)
    # incurs strong alternative-word suppression on the universal label.
    n_sole = jnp.sum(jnp.where(fc == 1.0, 1.0, 0.0))
    has_unique_sole = jnp.where(n_sole == 1.0, 1.0, 0.0)
    has_other_word = jnp.minimum(jnp.maximum(fc[None, :] - 1.0, 0.0), 1.0)
    alt_excess = (
        0.50 * fc_excess
        + 0.50 * jnp.maximum(fc[None, :] - 2.0, 0.0)
        + 2.50 * has_unique_sole * has_other_word
    )
    is_univ = jnp.where(feat_ext >= 4.0, 1.0, 0.0) * real_mask
    alt_avail = ctx.is_sink[:, None] * ctx.lex + real_mask * (real_lex / (1.0 + 0.50 * is_univ * alt_excess))

    return real_mask, real_lex, n_obj, prior, avail, alt_avail


def listener(params, ctx):
    _, real_lex, n_obj, prior, avail, alt_avail = _scene_components(params, ctx)

    prag = L2(params["alpha"], avail, prior)[ctx.utterance]
    lit_w = alt_avail[ctx.utterance]
    lit0 = lit_w / jnp.maximum(jnp.sum(lit_w), EPS)

    heard_ext = jnp.sum(real_lex[ctx.utterance])
    overext = jnp.maximum(heard_ext - 2.0, 0.0)
    is_sub_univ = jnp.where((heard_ext > 2.0) & (heard_ext < n_obj), 1.0, 0.0)
    ret_l0 = 0.29 * overext + 0.07 * is_sub_univ
    ret_prior = 0.0042 * overext
    heard = (1.0 - ret_l0 - ret_prior) * prag + ret_l0 * lit0 + ret_prior * prior

    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    real_mask, _, _, prior, avail, alt_avail = _scene_components(params, ctx)
    l1_tab = L1(params["alpha"], avail, prior)
    l0_tab = alt_avail / jnp.maximum(jnp.sum(alt_avail, axis=1, keepdims=True), EPS)
    eff_l = 0.75 * l1_tab + 0.25 * l0_tab
    w = real_mask * alt_avail * jnp.exp(params["alpha"] * jnp.log(eff_l + EPS))
    col_sum = jnp.sum(w, axis=0, keepdims=True)
    indifferent = real_mask / jnp.maximum(jnp.sum(real_mask), EPS)
    return jnp.where(col_sum > 0, w / jnp.maximum(col_sum, EPS), indifferent)
