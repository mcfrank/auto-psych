import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_lex": dist.LogNormal(0.0, 0.5),
}


@memo
def L0[u: UTT, r: OBJ](base_lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(base_lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, base_lex: ..., soft_lex: ..., spk_cost: ..., prior: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(base_lex, u, r)
            * exp(
                alpha * log(L0[u, r](base_lex) + {EPS})
                - at(spk_cost, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r] * at(soft_lex, u, r))
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, base_lex: ..., soft_lex: ..., spk_cost: ..., prior: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(base_lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, base_lex, soft_lex, spk_cost, prior) + {EPS})
                - at(spk_cost, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r] * at(soft_lex, u, r))
    return Pr[listener.r == r]


def compute_lexical_terms(ctx, w_lex):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist_mat = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist_mat.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist_mat) * (1.0 - eye)

    copy_count = jnp.sum(jnp.where(dist_mat == 0.0, 1.0, 0.0), axis=1)
    is_clone = jnp.where(copy_count > 1.0, 1.0, 0.0)
    is_singleton = 1.0 - is_clone

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
    is_empty = jnp.where((fc == 0.0) & (max_fc > 0.0), 1.0, 0.0)
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
    has_false_min_tier = jnp.where((min_fc > 0.0) & (min_true_fc > min_fc), 1.0, 0.0)

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

    n_min_feats = jnp.sum(is_min_utt, axis=0, keepdims=True)
    min_n_min_feats = jnp.max(is_min_utt * n_min_feats, axis=1, keepdims=True)
    gap_utt = jnp.maximum(fc[None, :] - min_true_fc - jnp.maximum(min_n_min_feats, 1.0), 0.0) / n_real_utt
    has_min_feat = jnp.max(
        is_min_utt + jnp.where(n_true == 1.0, real_lex, 0.0),
        axis=0,
    )

    has_utt_comp = jnp.where((min_cloned > 0) | (max_cloned > 0) | (has_inter_utt > 0), 1.0, 0.0)
    is_sole_false_singleton = (
        real_words
        * (1.0 - real_lex)
        * is_singleton[None, :]
        * jnp.where((n_true > 1.0) & (n_false == 1.0), 1.0, 0.0)
        * jnp.where(
            (is_min_fc[None, :] > 0) & (min_fc > 0.0),
            jnp.where((min_fc < min_true_fc) | (min_cloned > 0) | (has_inter_utt > 0), 1.0, 0.0),
            has_utt_comp,
        )
    )
    false_dist = (
        n_false
        + jnp.maximum(min_true_fc - fc[None, :], 1.0 + (1.0 - has_tier_span))
        + is_empty[None, :] * (1.0 - has_inter_utt) * jnp.where(min_true_fc == 1.0, 1.0, 0.0)
        - (1.0 - is_min_fc[None, :])
        * (min_cloned + jnp.where(fc[None, :] == max_true_fc, max_cloned, 0.0))
        - (min_cloned + max_cloned) / n_obj
    )
    base_lex = ctx.lex + is_sole_false_singleton * jnp.exp(-w_lex * false_dist)

    spk_cost = w_lex * (
        (real_lex @ sim)
        + is_inter_utt * (1.0 - (n_min_feats - 1.0) * is_singleton[None, :] / n_real_utt)
        + is_incomplete_utt
        * is_singleton[None, :]
        * (
            1.0
            - is_max_utt
            * min_cloned
            * (1.0 - has_inter_utt)
            * jnp.where(min_true_fc == 1.0, 1.0, 0.0)
        )
        / n_real_utt
    )

    multi_min_utt = jnp.where(
        jnp.sum(is_min_utt * is_singleton[None, :], axis=1, keepdims=True) > 1.0,
        1.0,
        0.0,
    )
    multi_max_utt = jnp.where(jnp.sum(is_max_utt, axis=1, keepdims=True) > 1.0, 1.0, 0.0)
    unshadowed_min_feats = jnp.maximum(n_min_feats - 1.0 - (fc[None, :] - n_min_feats), 0.0)
    truth_grade = (
        is_all_utt * (min_cloned + gap_utt) / n_obj
        + is_min_utt
        * is_singleton[None, :]
        * (
            unshadowed_min_feats * has_false_min_tier
            + (1.0 - has_uniq_feat[None, :]) * multi_min_utt * max_cloned
        )
        / n_real_utt
        - is_incomplete_utt
        * is_singleton[None, :]
        * (
            is_max_utt * (1.0 - has_uniq_feat[None, :]) * multi_max_utt * (1.0 - max_cloned)
            + (1.0 - has_min_feat[None, :]) * jnp.where(n_true > 1.0 + 1.0, 1.0, 0.0)
        )
        / n_real_utt
        - is_inter_utt * is_singleton[None, :] * (1.0 - has_min_feat[None, :]) / n_real_utt
    )
    soft_lex = jnp.exp(w_lex * truth_grade)

    baseline_crowd = (
        jnp.sum(sim, axis=1)
        + is_clone / n_obj
        - ctx.valence * fc
        + ctx.grayscale / n_obj
    )
    word_prior = softmax_prior(-w_lex * baseline_crowd)
    prior_choice = softmax_prior(
        -w_lex
        * (
            baseline_crowd
            + is_empty
            * is_singleton
            * (1.0 - ctx.has_familiarization)
            * (1.0 - jnp.abs(ctx.valence))
            * jnp.where(n_real_utt >= n_obj, 1.0, 0.0)
            / n_obj
            - ctx.familiarization
            + is_inter_scene
            * (1.0 - ctx.has_familiarization)
            * (1.0 - jnp.abs(ctx.valence))
            * (n_obj - 1.0 - min_scene_cloned * (1.0 + is_singleton) - inter_scene_cloned)
            / n_obj
            - (
                is_max_fc
                * (
                    1.0
                    - inter_scene_cloned * is_all_feats * jnp.where(min_fc > 0.0, 1.0, 0.0)
                    + jnp.where(min_fc == 0.0, inter_scene_cloned * (1.0 - is_all_feats), 0.0)
                )
                + is_all_feats * (has_inter_scene * is_singleton + min_scene_cloned + inter_scene_cloned)
            )
            / n_obj
        )
    )
    return base_lex, soft_lex, spk_cost, word_prior, prior_choice


def listener(params, ctx):
    base_lex, soft_lex, spk_cost, word_prior, prior_choice = compute_lexical_terms(
        ctx, params["w_lex"]
    )
    heard = L2(
        params["alpha"], base_lex, soft_lex, spk_cost, word_prior
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior_choice, heard)
