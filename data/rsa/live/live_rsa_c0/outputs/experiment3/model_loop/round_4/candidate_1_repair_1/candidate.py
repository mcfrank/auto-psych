"""Scalar-prototype population listener: a heterogeneous population mixture of
parsimonious scalar pragmatic listeners and prototypical maximizing listeners under
capacity-limited visual attention."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

MODES = jnp.arange(2)

PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "w_proto": dist.Normal(1.0, 1.0),
    "logit_p_pragmatic": dist.Normal(0.5, 1.0),
    "logit_att_leak": dist.Normal(-2.0, 1.0),
    "logit_p_base": dist.Normal(2.0, 1.0),
    "w_load": dist.Normal(0.5, 0.5),
    "w_popout": dist.Normal(1.0, 0.5),
    "w_candidate": dist.Normal(0.5, 0.5),
    "w_inhibit_leak": dist.Normal(0.0, 1.0),
    "w_typical": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., focal: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(focal, r) + {EPS})
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](
    alpha, w_confusion, lex: ..., focal: ..., focal_prior: ..., confusion: ...
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(focal_prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * L0[u, r](lex, focal)
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L_pragmatic[u: UTT, r: OBJ](
    alpha,
    w_confusion,
    p_attend,
    lex: ...,
    focal: ...,
    focal_prior: ...,
    confusion: ...,
    capture_prior: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(focal_prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * L1[u, r](alpha, w_confusion, lex, focal, focal_prior, confusion)
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(
        att_mode in MODES, wpp=p_attend if att_mode == 1 else (1.0 - p_attend)
    )
    listener: chooses(
        r in OBJ,
        wpp=Pr[speaker.r == r] if att_mode == 1 else vec(capture_prior, r),
    )
    return Pr[listener.r == r]


@memo
def L_proto[u: UTT, r: OBJ](
    w_proto,
    p_attend,
    lex: ...,
    proto_score: ...,
    capture_prior: ...,
):
    listener: knows(u)
    listener: chooses(
        att_mode in MODES, wpp=p_attend if att_mode == 1 else (1.0 - p_attend)
    )
    listener: chooses(
        r in OBJ,
        wpp=(at(lex, u, r) * exp(w_proto * vec(proto_score, r)) + {EPS})
        if att_mode == 1
        else vec(capture_prior, r),
    )
    return Pr[listener.r == r]


def compute_confusion(ctx):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist_mat = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist_mat.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist_mat) * (1.0 - eye)

    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    return real_lex @ sim


def compute_typicality(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_obj = ctx.lex.shape[1]
    k_u = jnp.sum(real_lex, axis=1, keepdims=True)
    shared_fraction = (k_u - 1.0) / jnp.maximum(1.0, n_obj - 1.0)
    return jnp.sum(real_lex * shared_fraction, axis=0)


def choice_probs(params, ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    effective_w_features = (
        params["w_features"] + ctx.valence * params["w_valence"]
    )

    typicality = compute_typicality(ctx)

    salience_logits = (
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + effective_w_features * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
        + params["w_typical"] * typicality
    )
    prior = softmax_prior(salience_logits)
    confusion = compute_confusion(ctx)

    real_matches = ctx.lex[ctx.utterance] * (1.0 - ctx.is_sink[ctx.utterance])
    matching_count = jnp.sum(real_matches)
    is_popout = jnp.where(matching_count == 1.0, 1.0, 0.0)
    candidate_load = jnp.maximum(0.0, matching_count - 1.0)

    n_obj = ctx.lex.shape[1]
    logit_p_attend = (
        params["logit_p_base"]
        + params["w_popout"] * is_popout
        - params["w_load"] * (n_obj - 2.0)
        - params["w_candidate"] * candidate_load
    )
    p_attend = jax.nn.sigmoid(logit_p_attend)
    p_pragmatic = jax.nn.sigmoid(params["logit_p_pragmatic"])

    att_leak = jax.nn.sigmoid(params["logit_att_leak"])
    focal_weights = jnp.where(
        ctx.is_prior > 0, 1.0, jnp.where(real_matches > 0, 1.0, att_leak)
    )
    focal = focal_weights / jnp.sum(focal_weights)
    focal_prior_weights = prior * focal_weights + 1e-6
    focal_prior = focal_prior_weights / jnp.sum(focal_prior_weights)

    inhibit_leak = jax.nn.softplus(params["w_inhibit_leak"]) * candidate_load
    inhibit_factor = jnp.where(real_matches > 0, 1.0, inhibit_leak)
    capture_weights = prior * inhibit_factor + 1e-6
    capture_prior = capture_weights / jnp.sum(capture_weights)

    feat_f = ctx.feature_count.astype(jnp.float32)
    proto_score = feat_f / jnp.maximum(1.0, jnp.max(feat_f))

    heard_pragmatic = L_pragmatic(
        params["alpha"],
        params["w_confusion"],
        p_attend,
        ctx.lex,
        focal,
        focal_prior,
        confusion,
        capture_prior,
    )[ctx.utterance]

    heard_proto = L_proto(
        params["w_proto"],
        p_attend,
        ctx.lex,
        proto_score,
        capture_prior,
    )[ctx.utterance]

    heard = p_pragmatic * heard_pragmatic + (1.0 - p_pragmatic) * heard_proto

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
