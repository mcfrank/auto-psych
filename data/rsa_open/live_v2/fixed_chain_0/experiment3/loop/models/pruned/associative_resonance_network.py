import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_inh": dist.LogNormal(0.0, 0.5),
}


@memo
def R0_obj[u: UTT, r: OBJ](assoc_lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(assoc_lex, u, r))
    return Pr[listener.r == r]


@memo
def R1_utt[u: UTT, r: OBJ](alpha, assoc_lex: ..., word_act: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(assoc_lex, u, r)
        * exp(
            alpha * log(R0_obj[u, r](assoc_lex) + {EPS})
            + at(word_act, u, r)
        ),
    )
    return Pr[speaker.u == u]


@memo
def R1_obj[u: UTT, r: OBJ](alpha, assoc_lex: ..., word_act: ..., obj_act: ..., dec_act: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(obj_act, r)
        * (R1_utt[u, r](alpha, assoc_lex, word_act) + {EPS})
        * at(dec_act, u, r),
    )
    return Pr[listener.r == r]


@memo
def R2_utt[u: UTT, r: OBJ](alpha, assoc_lex: ..., word_act: ..., obj_act: ..., dec_act: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(assoc_lex, u, r)
        * exp(
            alpha * log(R1_obj[u, r](alpha, assoc_lex, word_act, obj_act, dec_act) + {EPS})
            + at(word_act, u, r)
        ),
    )
    return Pr[speaker.u == u]


@memo
def R2_obj[u: UTT, r: OBJ](alpha, assoc_lex: ..., word_act: ..., obj_act: ..., dec_act: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(obj_act, r)
        * (R2_utt[u, r](alpha, assoc_lex, word_act, obj_act, dec_act) + {EPS})
        * at(dec_act, u, r),
    )
    return Pr[listener.r == r]


def compute_network_terms(ctx, w_inh):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist_mat = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist_mat.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist_mat) * (1.0 - eye)

    copy_count = jnp.sum(jnp.where(dist_mat == 0.0, 1.0, 0.0), axis=1)
    is_clone = jnp.where(copy_count > 1.0, 1.0, 0.0)
    is_singleton = 1.0 - is_clone
    unique_sim = sim / copy_count[None, :]

    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_real_utt = jnp.sum(1.0 - ctx.is_sink)
    n_true = jnp.sum(real_lex, axis=1, keepdims=True)
    n_false = n_obj - n_true
    has_uniq_feat = jnp.max(
        jnp.where(n_true == 1.0, real_lex, 0.0),
        axis=0,
    )

    fc = ctx.feature_count
    min_fc = jnp.min(fc)
    max_fc = jnp.max(fc)
    is_zero_fc = jnp.where(fc == 0.0, 1.0, 0.0)
    is_min_fc = jnp.where((fc == min_fc) & (max_fc > min_fc), 1.0, 0.0)
    is_inter_scene = jnp.where((fc > min_fc) & (fc < max_fc), 1.0, 0.0)
    has_inter_scene = jnp.max(is_inter_scene)
    is_max_fc = jnp.where((fc == max_fc) & (max_fc > min_fc), 1.0, 0.0)
    is_all_feats = jnp.where((fc == n_real_utt) & (max_fc > min_fc), 1.0, 0.0)
    min_scene_cloned = jnp.max(is_min_fc * is_clone)
    inter_scene_cloned = jnp.max(is_inter_scene * is_clone)

    masked_min_fc = jnp.where(real_lex > 0, fc[None, :], max_fc + 1.0)
    min_true_fc = jnp.min(masked_min_fc, axis=1, keepdims=True)
    masked_max_fc = jnp.where(real_lex > 0, fc[None, :], 0.0)
    max_true_fc = jnp.max(masked_max_fc, axis=1, keepdims=True)
    has_tier_span = jnp.where(max_true_fc > min_true_fc, 1.0, 0.0)

    is_min_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] == min_true_fc) & (max_true_fc > min_true_fc),
        1.0,
        0.0,
    )
    is_inter_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] > min_true_fc) & (fc[None, :] < max_true_fc),
        1.0,
        0.0,
    )
    is_incomplete_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] > min_true_fc) & (fc[None, :] < n_real_utt),
        1.0,
        0.0,
    )
    is_max_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] == max_true_fc) & (max_true_fc > min_true_fc),
        1.0,
        0.0,
    )
    is_all_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] == n_real_utt) & (max_true_fc > min_true_fc),
        1.0,
        0.0,
    )
    min_cloned = jnp.max(is_min_utt * is_clone[None, :], axis=1, keepdims=True)
    max_cloned = jnp.max(is_max_utt * is_clone[None, :], axis=1, keepdims=True)
    has_inter_utt = jnp.max(is_inter_utt, axis=1, keepdims=True)
    gap_utt = jnp.maximum(fc[None, :] - min_true_fc - 1.0, 0.0) / n_real_utt

    is_false_below = (
        real_words
        * (1.0 - real_lex)
        * is_min_fc[None, :]
        * is_singleton[None, :]
        * jnp.where((min_fc > 0.0) & (min_fc < min_true_fc), 1.0, 0.0)
        * jnp.where((n_true > 1.0) & (n_false == 1.0), 1.0, 0.0)
    )
    is_false_same_one = (
        real_words
        * (1.0 - real_lex)
        * is_min_fc[None, :]
        * is_singleton[None, :]
        * jnp.where((min_fc == 1.0) & (min_true_fc == 1.0), 1.0, 0.0)
        * has_tier_span
        * jnp.where((n_true > 1.0) & (n_false == 1.0), 1.0, 0.0)
    )
    assoc_lex = (
        ctx.lex
        + is_false_below
        * jnp.exp(
            -w_inh
            * (
                n_false
                + 1.0
                + (1.0 - has_tier_span)
                + (min_true_fc - 1.0 - min_fc)
                - has_inter_utt / n_obj
            )
        )
        + is_false_same_one
        * jnp.exp(-w_inh * (n_obj + 1.0 - min_cloned))
    )

    multi_min_singleton = jnp.where(
        jnp.sum(is_min_utt * is_singleton[None, :], axis=1, keepdims=True) > 1.0,
        1.0,
        0.0,
    )
    sim_utt = jnp.where(
        multi_min_singleton * max_cloned > 0,
        (real_lex @ unique_sim) + (copy_count[None, :] - 1.0) * real_lex,
        real_lex @ sim,
    )

    inh_cost = w_inh * (
        sim_utt
        + is_inter_utt * (1.0 + has_inter_utt / n_obj)
        + is_incomplete_utt * is_singleton[None, :] / n_real_utt
    )
    word_act = -inh_cost

    multi_max_utt = jnp.where(jnp.sum(is_max_utt, axis=1, keepdims=True) > 1.0, 1.0, 0.0)
    proto_span = (
        min_cloned
        + gap_utt * (1.0 - has_inter_utt * (1.0 - min_cloned))
    )
    dec_score = (
        is_all_utt * proto_span / n_obj
        + is_min_utt
        * (min_true_fc - 1.0)
        * (1.0 - has_inter_utt)
        * (1.0 - is_singleton[None, :] * (1.0 - max_cloned) * jnp.where((min_fc > 0.0) & (n_false > 1.0), 1.0, 0.0))
        / (n_obj * n_real_utt)
        - is_incomplete_utt
        * is_max_utt
        * is_singleton[None, :]
        * (1.0 - has_uniq_feat[None, :])
        * multi_max_utt
        * (1.0 - max_cloned)
        / n_real_utt
    )
    dec_act = jnp.exp(w_inh * dec_score)

    baseline_crowd = (
        jnp.sum(sim, axis=1)
        + is_clone / n_obj
        - ctx.valence * fc
        + ctx.grayscale / n_obj
    )
    obj_act = softmax_prior(-w_inh * baseline_crowd)
    prior_choice = softmax_prior(
        -w_inh
        * (
            baseline_crowd
            - ctx.familiarization
            + is_zero_fc * is_singleton * has_inter_scene / n_obj
            + is_inter_scene
            * (1.0 - ctx.has_familiarization)
            * (1.0 - jnp.abs(ctx.valence))
            * (n_obj - 1.0 - min_scene_cloned * (1.0 + is_singleton) - inter_scene_cloned)
            / n_obj
            - (
                is_max_fc * (1.0 - inter_scene_cloned)
                + is_all_feats * (has_inter_scene * is_singleton + min_scene_cloned + inter_scene_cloned)
            )
            / n_obj
        )
    )
    return assoc_lex, word_act, obj_act, dec_act, prior_choice


def listener(params, ctx):
    assoc_lex, word_act, obj_act, dec_act, prior_choice = compute_network_terms(
        ctx, params["w_inh"]
    )
    heard = R2_obj(
        params["alpha"], assoc_lex, word_act, obj_act, dec_act
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior_choice, heard)


def speaker(params, ctx):
    assoc_lex, word_act, obj_act, dec_act, _ = compute_network_terms(
        ctx, params["w_inh"]
    )
    raw = R2_utt(
        params["alpha"], assoc_lex, word_act, obj_act, dec_act
    )
    real = (1.0 - ctx.is_sink)[:, None]
    weights = raw * real
    total = jnp.sum(weights, axis=0, keepdims=True)
    indifferent = jnp.broadcast_to(real / jnp.sum(real), weights.shape)
    return jnp.where(total > 0, weights / jnp.where(total > 0, total, 1.0), indifferent)
