"""Aspect QUD listener.

In reference games, listeners interpret utterances by reasoning about a speaker
who communicates an informative aspect of the target rather than its complete identity.
When selecting an aspect to address as the question under discussion, the speaker chooses
among the target's true properties with a preference for distinctive features, and selects
an utterance to communicate that aspect to a literal listener. On uninformative displays
without an informative word, visual attention defaults to contextually unique singletons.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_aspect": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., aspect_weight: ..., qud_utility: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(g in UTT, wpp=at(aspect_weight, g, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * at(qud_utility, u, g)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_singleton(ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    return jnp.where(duplicate_count == 1, 1.0, 0.0)


def compute_aspect_weights(ctx, w_aspect):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_obj = ctx.lex.shape[1]
    ext = jnp.maximum(jnp.sum(real_lex, axis=1), 1.0)
    info = jnp.log(n_obj / ext) * (1.0 - ctx.is_sink)
    aspect_weight = ctx.lex * jnp.exp(w_aspect * info)[:, None]
    return aspect_weight


def compute_qud_utility(ctx):
    l0 = L0(ctx.lex)
    p_aspect = l0 @ ctx.lex.T
    return jnp.log(jnp.clip(p_aspect, 0.01, 1.0))


def listener(params, ctx):
    aspect_weight = compute_aspect_weights(ctx, params["w_aspect"])
    qud_utility = compute_qud_utility(ctx)
    heard = L1(params["alpha"], ctx.lex, aspect_weight, qud_utility)[ctx.utterance]
    is_singleton = compute_singleton(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    return jnp.where(ctx.is_prior > 0, prior, heard)
