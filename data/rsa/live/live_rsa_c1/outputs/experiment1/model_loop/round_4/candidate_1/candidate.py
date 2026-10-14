"""Literal-pragmatic mixture listener.

Listeners in reference games are cognitively heterogeneous: a proportion of listeners
interpret utterances literally, selecting among truth-conditional matches according to
visual attention without recursive speaker reasoning, while the remaining proportion
engage in depth-2 pragmatic reasoning by inverting an informative speaker who simulates
a pragmatic listener. Observed population choices reflect the mixture of these two
reasoning depths.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "p_literal": dist.Beta(1.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., attention: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(attention, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., attention: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(attention, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, attention) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., attention: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(attention, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, attention) + {EPS})),
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

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    attention = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
    )

    l2_heard = L2(params["alpha"], ctx.lex, attention)[ctx.utterance]
    l0_heard = L0(ctx.lex, attention)[ctx.utterance]

    heard = (1.0 - params["p_literal"]) * l2_heard + params["p_literal"] * l0_heard
    choice = jnp.where(ctx.is_prior > 0, attention, heard)
    return with_lapse(choice, params["lapse"])
