"""Limited attention listener with referential ambiguity modulation and competitor confusion avoidance.

Refining limited_attention_listener_2 by incorporating referential ambiguity modulation into the capacity-limited
attention mechanism: listeners balance top-down pragmatic reasoning against bottom-up visual attentional capture,
with attentional engagement modulated not only by display size and pop-out, but by the communicative ambiguity
of the referring expression. When an expression applies to multiple candidate referents, top-down attentional
control is diluted, increasing the probability of bottom-up visual capture by salient objects in the scene.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

MODES = jnp.arange(2)

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 1.0),
    "logit_p_base": dist.Normal(2.0, 1.0),
    "w_load": dist.Normal(0.5, 0.5),
    "w_popout": dist.Normal(1.0, 0.5),
    "w_ambiguity": dist.Normal(0.0, 1.0),
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
def L_attention[u: UTT, r: OBJ](alpha, w_confusion, p_attend, lex: ..., attention: ..., confusion: ...):
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
    listener: chooses(mode in MODES, wpp=p_attend if mode == 1 else (1.0 - p_attend))
    listener: chooses(
        r in OBJ,
        wpp=Pr[speaker.r == r] if mode == 1 else vec(attention, r),
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
    ambiguity = jnp.maximum(0.0, matching_count - 1.0)

    n_obj = ctx.lex.shape[1]
    logit_p_attend = (
        params["logit_p_base"]
        + params["w_popout"] * is_popout
        - params["w_load"] * (n_obj - 2.0)
        - params["w_ambiguity"] * ambiguity
    )
    p_attend = jax.nn.sigmoid(logit_p_attend)

    heard = L_attention(params["alpha"], params["w_confusion"], p_attend, ctx.lex, attention, confusion)[ctx.utterance]
    probs = jnp.where(ctx.is_prior > 0, attention, heard)
    return with_lapse(probs, params["lapse"])
