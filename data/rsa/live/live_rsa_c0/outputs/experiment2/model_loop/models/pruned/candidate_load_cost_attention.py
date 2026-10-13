"""Pragmatic listener at depth 2 with capacity-limited attention, expected communicative success,
competitor confusion avoidance, utterance specificity costs, and referential candidate load attentional modulation.

Refining success_confusion_cost_attention by modulating capacity-limited visual attention by referential candidate load:
when an utterance applies to multiple candidate referents in the visual scene, the cognitive burden of evaluating
competing referents dilutes top-down attentional control, increasing the probability of bottom-up visual attentional
capture by salient contextual objects without communicative verification.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

MODES = jnp.arange(2)

PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "c_spec": dist.Normal(0.0, 1.0),
    "logit_p_base": dist.Normal(2.0, 1.0),
    "w_load": dist.Normal(0.5, 0.5),
    "w_popout": dist.Normal(1.0, 0.5),
    "w_candidate": dist.Normal(0.5, 0.5),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., prior: ..., confusion: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * (L0[u, r](lex) - vec(cost, u))
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L_attention[u: UTT, r: OBJ](
    alpha, w_confusion, p_attend, lex: ..., prior: ..., confusion: ..., cost: ...
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * (L1[u, r](alpha, w_confusion, lex, prior, confusion, cost) - vec(cost, u))
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(mode in MODES, wpp=p_attend if mode == 1 else (1.0 - p_attend))
    listener: chooses(
        r in OBJ,
        wpp=Pr[speaker.r == r] if mode == 1 else vec(prior, r),
    )
    return Pr[listener.r == r]


def compute_confusion(ctx):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)

    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    return real_lex @ sim


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

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + effective_w_features * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    confusion = compute_confusion(ctx)

    n_objects = ctx.lex.shape[1]
    extension = jnp.sum(real_lex, axis=1) / n_objects
    specificity = (1.0 - extension) * (1.0 - ctx.is_sink)
    cost = params["c_spec"] * specificity

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

    heard = L_attention(
        params["alpha"],
        params["w_confusion"],
        p_attend,
        ctx.lex,
        prior,
        confusion,
        cost,
    )[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
