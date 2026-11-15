import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_anchor": dist.LogNormal(0.0, 0.5),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., spec_cost: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - at(spec_cost, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., spec_cost: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, lex, spec_cost, prior) + {EPS})
                - at(spec_cost, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_specificity_and_prior(ctx, w_anchor):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist_mat = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist_mat.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist_mat) * (1.0 - eye)
    is_clone = jnp.where(jnp.sum(dist_mat == 0.0, axis=1) > 1.0, 1.0, 0.0)

    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_real_utt = jnp.sum(1.0 - ctx.is_sink)

    fc = ctx.feature_count
    min_fc = jnp.min(fc)
    max_fc = jnp.max(fc)
    is_inter_scene = jnp.where((fc > min_fc) & (fc < max_fc), 1.0, 0.0)
    has_inter_scene = jnp.max(is_inter_scene)
    has_clone_scene = jnp.max(is_clone)
    is_max_fc = jnp.where((fc == max_fc) & (max_fc > min_fc), 1.0, 0.0)
    is_all_feats = jnp.where((fc == n_real_utt) & (max_fc > min_fc), 1.0, 0.0)

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
    min_cloned_utt = jnp.where(jnp.sum(is_min_utt, axis=1, keepdims=True) > 1.0, 1.0, 0.0)
    is_unique_proto_utt = (
        real_lex * is_all_feats[None, :] * (1.0 - is_clone[None, :])
    )

    spec_cost = w_anchor * (
        (real_lex @ sim)
        + is_inter_utt
        + is_incomplete_utt * (1.0 - is_clone[None, :]) / n_real_utt
        - is_unique_proto_utt * min_cloned_utt / n_obj
    )
    baseline_crowd = (
        jnp.sum(sim, axis=1)
        + is_clone * (1.0 - is_inter_scene) / n_obj
        - ctx.valence * fc
        + ctx.grayscale / n_obj
    )
    word_prior = softmax_prior(
        -w_anchor
        * (
            baseline_crowd
            - is_all_feats * (max_fc - min_fc - 1.0) / (n_obj * n_real_utt)
        )
    )
    prior_choice = softmax_prior(
        -w_anchor
        * (
            baseline_crowd
            - ctx.familiarization
            + is_inter_scene
            * (1.0 - ctx.has_familiarization)
            * (1.0 - jnp.abs(ctx.valence))
            * (n_obj - 1.0)
            / n_obj
            - (
                is_max_fc
                + is_all_feats * (1.0 - is_clone) * jnp.maximum(has_inter_scene, has_clone_scene)
            )
            / n_obj
        )
    )
    return spec_cost, word_prior, prior_choice


def listener(params, ctx):
    spec_cost, word_prior, prior_choice = compute_specificity_and_prior(
        ctx, params["w_anchor"]
    )
    heard = L2(params["alpha"], ctx.lex, spec_cost, word_prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior_choice, heard)
