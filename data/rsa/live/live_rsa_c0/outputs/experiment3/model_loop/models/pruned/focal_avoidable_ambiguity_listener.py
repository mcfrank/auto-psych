"""Focal avoidable ambiguity listener: pragmatic listener under capacity-limited
visual attention with focal candidate contrast set mentalizing and avoidable ambiguity costs.

Refining avoidable_ambiguity_speaker by incorporating the focal candidate attention
component from focal_candidate_attention_listener: listeners interpret referring
expressions under capacity-limited visual attention that restricts counterfactual
communicative mentalizing to the active focal set of candidate referents matching
the heard word. When evaluating what a speaker would have said, listeners mentally
simulate alternative expressions only with respect to distinguishing candidates
within this attended contrast set rather than tracking distractors already ruled out
by the literal word. Consequently, distinguishing features that isolate a competitor
within the focal candidate set strongly preempt that competitor, elevating choices of
multi-feature pragmatic targets over all-feature competitors in four-object displays
(four_object_multifeature_target_vs_all_feature_rate and
four_object_two_feature_target_vs_three_feature_competitor_rate) and targets possessing
unique alternative features (unique_alternative_target_choice_rate), while preserving
avoidable ambiguity penalties when multiple matching candidates remain.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

MODES = jnp.arange(2)

PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "c_avoid": dist.Normal(0.0, 1.0),
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
    alpha,
    w_confusion,
    lex: ...,
    focal: ...,
    focal_prior: ...,
    confusion: ...,
    avoidable: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(focal_prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * (L0[u, r](lex, focal) - at(avoidable, u, r))
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](
    alpha,
    w_confusion,
    p_attend,
    lex: ...,
    focal: ...,
    focal_prior: ...,
    confusion: ...,
    capture_prior: ...,
    avoidable: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(focal_prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * (
                    L1[u, r](
                        alpha,
                        w_confusion,
                        lex,
                        focal,
                        focal_prior,
                        confusion,
                        avoidable,
                    )
                    - at(avoidable, u, r)
                )
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(
        mode in MODES, wpp=p_attend if mode == 1 else (1.0 - p_attend)
    )
    listener: chooses(
        r in OBJ,
        wpp=Pr[speaker.r == r] if mode == 1 else vec(capture_prior, r),
    )
    return Pr[listener.r == r]


def compute_confusion(ctx):
    feats = jnp.vstack(
        [ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]]
    )
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)

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


def compute_avoidable(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    k_u = jnp.sum(real_lex, axis=1)
    n_obj = ctx.lex.shape[1]
    ambiguity = (k_u - 1.0) / jnp.maximum(1.0, n_obj - 1.0) * (1.0 - ctx.is_sink)
    is_unique_word = (k_u == 1.0) * (1.0 - ctx.is_sink)
    has_unique = jnp.max(real_lex * is_unique_word[:, None], axis=0)
    avoidable = real_lex * ambiguity[:, None] * has_unique[None, :]
    return avoidable


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

    avoidable_matrix = compute_avoidable(ctx)
    avoidable = params["c_avoid"] * avoidable_matrix

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

    heard = L2(
        params["alpha"],
        params["w_confusion"],
        p_attend,
        ctx.lex,
        focal,
        focal_prior,
        confusion,
        capture_prior,
        avoidable,
    )[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
