"""Pragmatic listener inverting a speaker under figural topic prominence.

Listeners evaluate candidate referents before hearing an utterance through figural
topic prominence: communicative expectations and visual attention are grounded in
visual feature presence (figure-ground segregation). Unadorned baseline objects serve
as the perceptual ground rather than informative topics, so visual pop-out applies
strictly to feature-bearing figures that possess distinguishing visual features.
Topic expectations concentrate on the continuum endpoints—the bare baseline contrast
and the maximally elaborated focal exemplar—while intermediate-complexity objects
suffer perceptual compression.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "epsilon": dist.Beta(1.0, 5.0),
    "w_feat_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_extremity": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(prior, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, epsilon, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            m in UTT,
            wpp=at(lex, m, r) * exp(alpha * log(L0[m, r](lex, prior) + {EPS})),
        ),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * ((1.0 - epsilon) * (u == m) + epsilon),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    duplicate_count = jnp.sum(pair_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)

    feat_singleton = is_singleton * (ctx.feature_count > 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    min_count = jnp.min(ctx.feature_count)
    max_count = jnp.max(ctx.feature_count)
    midpoint = 0.5 * (min_count + max_count)
    extremity = jnp.abs(ctx.feature_count - midpoint)

    prior = softmax_prior(
        params["w_feat_singleton"] * feat_singleton
        + params["w_distinct"] * distinct
        + params["w_extremity"] * extremity
        + params["w_features"] * ctx.feature_count
    )

    heard = L1(
        params["alpha"], params["epsilon"], ctx.lex, prior
    )[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
