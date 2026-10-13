"""Sampled inference listener: resource-bounded pragmatic listener using
stochastic candidate sampling and pairwise contrastive mentalizing.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "logit_p_sample": dist.Normal(1.0, 1.0),
    "logit_att_leak": dist.Normal(-2.5, 1.0),
    "w_load": dist.Normal(0.5, 0.5),
    "w_popout": dist.Normal(1.0, 0.5),
    "w_candidate": dist.Normal(0.5, 0.5),
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
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., prior: ..., confusion: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * L0[u, r](lex) - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def S2[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., prior: ..., confusion: ...):
    speaker: given(r in OBJ, wpp=1)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r)
        * exp(
            alpha * L1[u, r](alpha, w_confusion, lex, prior, confusion)
            - w_confusion * at(confusion, u, r)
        ),
    )
    return Pr[speaker.u == u]


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

    att_leak = jax.nn.sigmoid(params["logit_att_leak"])
    prop_weights = prior * jnp.where(
        ctx.is_prior > 0, 1.0, jnp.where(real_matches > 0, 1.0, att_leak)
    ) + 1e-6
    q = prop_weights / jnp.sum(prop_weights)

    n_obj = ctx.lex.shape[1]
    logit_p_sample = (
        params["logit_p_sample"]
        + params["w_popout"] * is_popout
        - params["w_load"] * (n_obj - 2.0)
        - params["w_candidate"] * candidate_load
    )
    p_sample = jax.nn.sigmoid(logit_p_sample)

    s2_all = S2(params["alpha"], params["w_confusion"], ctx.lex, prior, confusion)
    w = s2_all[ctx.utterance]

    # Pairwise contrastive tournament between sampled candidates
    w_safe = w + 1e-4
    denom = w_safe[:, None] + w_safe[None, :]
    pair_prob = w_safe[:, None] / denom
    eye = jnp.eye(n_obj)
    pair_prob = pair_prob * (1.0 - eye)

    comp = 2.0 * jnp.sum(q[None, :] * pair_prob, axis=1)
    p2 = q * (q + comp)

    p_sampled = (1.0 - p_sample) * q + p_sample * p2

    heard = p_sampled
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
