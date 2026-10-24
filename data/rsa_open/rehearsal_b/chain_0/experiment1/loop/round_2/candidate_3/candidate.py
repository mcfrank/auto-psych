"""Graded truth listener.

People interpret referring expressions using graded semantic truth values
that scale inversely with a referent's feature complexity. When hearing a word,
listeners evaluate an object's applicability not as a rigid binary match,
but as graded truth that diminishes as the object possesses more competing
visual features. On uninformative trials without an informative word,
visual attention defaults to contextually unique singletons.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](soft_lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(soft_lex, u, r) + {EPS})
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, soft_lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(soft_lex, u, r)
            * exp(alpha * log(L0[u, r](soft_lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_singleton(ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    return jnp.where(duplicate_count == 1, 1.0, 0.0)


def compute_graded_lex(ctx, gamma):
    fc = jnp.maximum(ctx.feature_count, 1.0)
    specificity = jnp.power(fc, -gamma)
    real_words = (1.0 - ctx.is_sink)[:, None]
    graded = ctx.lex * specificity[None, :] * real_words
    return jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, graded)


def listener(params, ctx):
    soft_lex = compute_graded_lex(ctx, params["gamma"])
    is_singleton = compute_singleton(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    heard = L1(params["alpha"], soft_lex, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
