"""Semantic information coverage listener.

Listeners interpret referring expressions through semantic information coverage,
weighting candidate referents by the proportion of the object's total contextual
feature information explained by the heard word. When a speaker uses a shared
or underspecified description, referents possessing unmentioned distinguishing
features are sharply discounted because cooperative speakers do not leave
highly informative properties unsaid. On uninformative displays without an
informative word, visual attention defaults to contextually unique singletons.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex_cov: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex_cov, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., lex_cov: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex_cov) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=vec(prior, r) * Pr[speaker.r == r])
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


def compute_coverage_lex(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_obj = ctx.lex.shape[1]
    ext = jnp.sum(real_lex, axis=1)
    info = (1.0 - ctx.is_sink) * jnp.log(
        (n_obj + 1.0) / (jnp.maximum(ext, 1.0) + 0.5)
    )
    obj_feat_info = real_lex * info[:, None]
    total_info = jnp.sum(obj_feat_info, axis=0)
    coverage = obj_feat_info / jnp.maximum(total_info[None, :], 1e-4)
    return jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, coverage)


def listener(params, ctx):
    is_singleton = compute_singleton(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    lex_cov = compute_coverage_lex(ctx)
    heard = L1(params["alpha"], ctx.lex, lex_cov, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
