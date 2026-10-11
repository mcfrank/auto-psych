"""Literal-pragmatic population mixture listener under capacity-limited attention.

Listeners in reference games belong to a heterogeneous population composed of
distinct cognitive types: literal listeners who choose among semantically valid
referents based on visual salience without computing speaker intentionality,
and pragmatic listeners who invert an informative speaker to derive scalar
implicatures. The overall communicative choice distribution reflects this mixture
of literal and pragmatic types within an attentive processing mode, balanced against
capacity-limited visual attentional capture under visual load.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 1.0),
    "logit_p_attend": dist.Normal(2.0, 1.0),
    "logit_p_pragmatic": dist.Normal(1.0, 1.0),
    "w_load": dist.Normal(0.5, 0.5),
    "w_popout": dist.Normal(1.0, 0.5),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., attention: ..., confusion: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(attention, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., attention: ..., confusion: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(attention, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, w_confusion, lex, attention, confusion) + {EPS})
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L_literal[u: UTT, r: OBJ](lex: ..., attention: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(attention, r))
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


def compute_salience(ctx, params):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    copy_count = jnp.sum(pair_dist == 0, axis=1)
    is_singleton = jnp.where(copy_count == 1, 1.0, 0.0)
    distinct = jnp.sum(diff, axis=(0, 2))

    salience_logits = (
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
    )
    return softmax_prior(salience_logits)


def choice_probs(params, ctx):
    attention = compute_salience(ctx, params)
    confusion = compute_confusion(ctx)

    real_matches = ctx.lex[ctx.utterance] * (1.0 - ctx.is_sink[ctx.utterance])
    matching_count = jnp.sum(real_matches)
    is_popout = jnp.where(matching_count == 1.0, 1.0, 0.0)

    n_obj = ctx.lex.shape[1]
    logit_p_attend = (
        params["logit_p_attend"]
        + params["w_popout"] * is_popout
        - params["w_load"] * (n_obj - 2.0)
    )
    p_attend = jax.nn.sigmoid(logit_p_attend)
    p_pragmatic = jax.nn.sigmoid(params["logit_p_pragmatic"])

    heard_pragmatic = L2(params["alpha"], params["w_confusion"], ctx.lex, attention, confusion)[ctx.utterance]
    heard_literal = L_literal(ctx.lex, attention)[ctx.utterance]
    heard_language = p_pragmatic * heard_pragmatic + (1.0 - p_pragmatic) * heard_literal

    heard = p_attend * heard_language + (1.0 - p_attend) * attention

    probs = jnp.where(ctx.is_prior > 0, attention, heard)
    return with_lapse(probs, params["lapse"])
