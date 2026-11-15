import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_lex": dist.LogNormal(0.0, 0.5),
}


@memo
def L0[u: UTT, r: OBJ](sem_l0: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(sem_l0, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, sem_l0: ..., sem_spk: ..., utt_cost: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(sem_spk, u, r)
            * exp(
                alpha * log(L0[u, r](sem_l0) + {EPS})
                - at(utt_cost, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, sem_l0: ..., sem_spk: ..., utt_cost: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(sem_spk, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, sem_l0, sem_spk, utt_cost, prior) + {EPS})
                - at(utt_cost, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_lexical_and_prior(ctx, w_lex):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist_mat = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist_mat.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist_mat) * (1.0 - eye)
    is_clone = jnp.where(jnp.sum(dist_mat == 0.0, axis=1) > 1.0, 1.0, 0.0)
    is_singleton = 1.0 - is_clone

    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_real_utt = jnp.sum(1.0 - ctx.is_sink)

    fc = ctx.feature_count
    min_fc = jnp.min(fc)
    max_fc = jnp.max(fc)
    is_min_fc = jnp.where((fc == min_fc) & (max_fc > min_fc), 1.0, 0.0)
    is_inter_scene = jnp.where((fc > min_fc) & (fc < max_fc), 1.0, 0.0)
    has_inter_scene = jnp.max(is_inter_scene)
    is_max_fc = jnp.where((fc == max_fc) & (max_fc > min_fc), 1.0, 0.0)
    is_all_feats = jnp.where((fc == n_real_utt) & (max_fc > min_fc), 1.0, 0.0)
    min_scene_cloned = jnp.max(is_min_fc * is_clone)

    masked_min_fc = jnp.where(real_lex > 0, fc[None, :], max_fc + 1.0)
    min_true_fc = jnp.min(masked_min_fc, axis=1, keepdims=True)
    masked_max_fc = jnp.where(real_lex > 0, fc[None, :], 0.0)
    max_true_fc = jnp.max(masked_max_fc, axis=1, keepdims=True)

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
    is_min_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] == min_true_fc) & (max_true_fc > min_true_fc),
        1.0,
        0.0,
    )
    is_max_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] == max_true_fc) & (max_true_fc > min_true_fc),
        1.0,
        0.0,
    )
    is_all_utt = is_max_utt * is_all_feats[None, :]
    min_cloned = jnp.max(is_min_utt * is_clone[None, :], axis=1, keepdims=True)
    gap_utt = jnp.maximum(fc[None, :] - min_true_fc - 1.0, 0.0) / n_real_utt

    feat_freq = jnp.sum(real_lex, axis=1, keepdims=True)
    has_unique_feat = jnp.max(real_lex * jnp.where(feat_freq == 1.0, 1.0, 0.0), axis=0)
    no_unique_feat = 1.0 - has_unique_feat
    has_min_singleton_utt = jnp.max(is_min_utt * is_singleton[None, :], axis=1, keepdims=True)

    # Graded literal semantics in L0: objects with unmentioned features and no unique feature are softer matches
    sem_l0 = ctx.lex * jnp.exp(
        -w_lex
        * (
            is_incomplete_utt * no_unique_feat[None, :]
            + is_min_utt * is_singleton[None, :] * has_unique_feat[None, :]
        )
        / n_real_utt
    )

    # Noisy lexical applicability in speaker: sparse minimal singletons can be noisily accessed
    is_false_min = (
        real_words
        * (1.0 - real_lex)
        * is_min_fc[None, :]
        * is_singleton[None, :]
        * jnp.where((min_fc > 0.0) & (min_fc < min_true_fc), 1.0, 0.0)
    )
    sem_spk = ctx.lex + is_false_min * jnp.exp(-w_lex * (n_obj - min_cloned))

    utt_cost = w_lex * (
        (real_lex @ sim)
        + is_inter_utt
        + is_incomplete_utt
        * (
            is_singleton[None, :]
            + (no_unique_feat[None, :] - is_singleton[None, :] * has_unique_feat[None, :] * has_min_singleton_utt)
            / n_obj
        )
        / n_real_utt
        - is_all_utt * (min_cloned * (1.0 - jnp.max(is_inter_utt, axis=1, keepdims=True)) + gap_utt) / n_obj
    )
    baseline_confusion = (
        jnp.sum(sim, axis=1)
        + is_clone / n_obj
        - ctx.valence * fc
        + ctx.grayscale / n_obj
    )
    word_prior = softmax_prior(-w_lex * baseline_confusion)
    is_unique_max = jnp.where(jnp.sum(is_max_fc) == 1.0, 1.0, 0.0)
    has_unique_inter = jnp.max(is_inter_scene * is_singleton)
    prior_choice = softmax_prior(
        -w_lex
        * (
            baseline_confusion
            - ctx.familiarization
            + is_inter_scene
            * (1.0 - ctx.has_familiarization)
            * (1.0 - jnp.abs(ctx.valence))
            * (n_obj - 1.0 - min_scene_cloned)
            / (n_obj + is_singleton * min_scene_cloned)
            - (
                is_max_fc * (is_all_feats + (1.0 - has_inter_scene * is_unique_max) * (1.0 - is_all_feats))
                + is_all_feats * (has_inter_scene * is_singleton + min_scene_cloned)
                + is_min_fc * is_singleton * has_unique_inter * is_unique_max / n_real_utt
            )
            / n_obj
        )
    )
    return sem_l0, sem_spk, utt_cost, word_prior, prior_choice


def listener(params, ctx):
    sem_l0, sem_spk, utt_cost, word_prior, prior_choice = compute_lexical_and_prior(
        ctx, params["w_lex"]
    )
    heard = L2(params["alpha"], sem_l0, sem_spk, utt_cost, word_prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior_choice, heard)
