"""Pragmatic listener reasoning with description exhaustivity.

Listeners interpret referring expressions through description exhaustivity,
assuming that cooperative speakers use single words to exhaustively describe
minimal referents and applying a discrete non-exhaustivity penalty to any candidate
referent possessing unmentioned features regardless of how many additional features
it has. Communicative speakers choose expressions by anticipating this
exhaustivity-sensitive listener, while spontaneous visual attention on uninformative
displays defaults to contextually unique singletons.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex_exh: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex_exh, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., lex_exh: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex_exh) + {EPS})),
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


def listener(params, ctx):
    is_singleton = compute_singleton(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)

    is_non_exhaustive = jnp.where(ctx.feature_count > 1, 1.0, 0.0)
    penalty = jnp.exp(-params["beta"] * is_non_exhaustive)
    lex_exh = jnp.where(
        ctx.is_sink[:, None] > 0,
        ctx.lex,
        ctx.lex * penalty[None, :],
    )

    heard = L1(params["alpha"], ctx.lex, lex_exh, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
