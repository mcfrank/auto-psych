import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior

PARAMS = {
    "w_diag": dist.LogNormal(0.0, 0.5),
    "w_crowd": dist.LogNormal(0.0, 0.5),
}


@memo
def L_heur[u: UTT, r: OBJ](sem_lex: ..., score: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(sem_lex, u, r) * exp(at(score, u, r)))
    return Pr[listener.r == r]


def compute_heuristic_scores(ctx, w_diag, w_crowd):
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
    has_uniq_feat = jnp.max(
        jnp.where(jnp.sum(real_lex, axis=1, keepdims=True) == 1.0, real_lex, 0.0),
        axis=0,
    )

    fc = ctx.feature_count
    min_fc = jnp.min(fc)
    max_fc = jnp.max(fc)
    is_min_fc = jnp.where((fc == min_fc) & (max_fc > min_fc), 1.0, 0.0)
    is_inter_scene = jnp.where((fc > min_fc) & (fc < max_fc), 1.0, 0.0)
    has_inter_scene = jnp.max(is_inter_scene)
    is_max_fc = jnp.where((fc == max_fc) & (max_fc > min_fc), 1.0, 0.0)
    is_all_feats = jnp.where((fc == n_real_utt) & (max_fc > min_fc), 1.0, 0.0)
    min_scene_cloned = jnp.max(is_min_fc * is_clone)
    inter_scene_cloned = jnp.max(is_inter_scene * is_clone)
    max_scene_cloned = jnp.max(is_max_fc * is_clone)

    masked_min_fc = jnp.where(real_lex > 0, fc[None, :], max_fc + 1.0)
    min_true_fc = jnp.min(masked_min_fc, axis=1, keepdims=True)
    masked_max_fc = jnp.where(real_lex > 0, fc[None, :], 0.0)
    max_true_fc = jnp.max(masked_max_fc, axis=1, keepdims=True)

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
    gap_utt = jnp.maximum(fc[None, :] - min_true_fc - jnp.where(min_true_fc == 1.0, 1.0, 0.0), 0.0) / n_real_utt

    is_false_min = (
        real_words
        * (1.0 - real_lex)
        * is_min_fc[None, :]
        * is_singleton[None, :]
        * jnp.where((min_fc > 0.0) & (min_fc < min_true_fc), 1.0, 0.0)
        * jnp.where(jnp.sum(real_lex, axis=1, keepdims=True) > 1.0, 1.0, 0.0)
    )
    sem_lex = ctx.lex + is_false_min * jnp.exp(
        -w_crowd * (n_obj - min_cloned - max_cloned - has_inter_utt)
    )

    word_crowd = (
        (real_lex @ sim)
        + is_inter_utt * (1.0 - has_uniq_feat[None, :])
        + is_incomplete_utt
        * is_singleton[None, :]
        * (1.0 + (1.0 - has_uniq_feat[None, :]) * (1.0 - max_cloned))
        / n_real_utt
        - is_all_utt * (min_cloned + gap_utt * min_cloned) / n_obj
    )

    baseline_crowd = (
        jnp.sum(sim, axis=1)
        + is_clone / n_obj
        - ctx.valence * fc
        + ctx.grayscale / n_obj
    )

    prior_choice = softmax_prior(
        -w_crowd
        * (
            baseline_crowd
            - ctx.familiarization
            + is_inter_scene
            * (1.0 - ctx.has_familiarization)
            * (1.0 - jnp.abs(ctx.valence))
            * (n_obj - 1.0 - min_scene_cloned * (1.0 + is_singleton))
            / n_obj
            - (
                is_max_fc
                * (1.0 - inter_scene_cloned)
                * (1.0 - max_scene_cloned * (1.0 - has_inter_scene))
                + is_all_feats * (has_inter_scene * is_singleton + min_scene_cloned + inter_scene_cloned)
            )
            / n_obj
        )
    )

    feat_ext = jnp.sum(sem_lex, axis=1, keepdims=True) + EPS
    feat_sal = sem_lex * jnp.exp(-w_diag * jnp.log(feat_ext) - w_crowd * word_crowd)
    obj_avail = feat_sal / (jnp.sum(feat_sal, axis=0, keepdims=True) + EPS)
    avail_score = jnp.log(obj_avail / (sem_lex + EPS) + EPS) - w_crowd * baseline_crowd[None, :]

    score = jnp.where(ctx.is_sink[:, None] > 0, 0.0, avail_score)
    return sem_lex, score, prior_choice


def listener(params, ctx):
    sem_lex, score, prior_choice = compute_heuristic_scores(
        ctx, params["w_diag"], params["w_crowd"]
    )
    heard = L_heur(sem_lex, score)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior_choice, heard)
