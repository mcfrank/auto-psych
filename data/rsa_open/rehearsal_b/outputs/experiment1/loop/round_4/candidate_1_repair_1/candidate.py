"""Literal-pragmatic listener mixture theory.

Listeners vary in their communicative reasoning depth: the population is a mixture
of literal listeners who interpret referring expressions according to literal
semantic truth, and pragmatic listeners who invert an informative speaker.
Communicative speakers choose words by expecting this heterogeneous population
of listeners. On uninformative displays without an informative word, visual
attention defaults to contextually unique singletons.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_literal": dist.Beta(1.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
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


def listener(params, ctx):
    is_singleton = compute_singleton(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)

    l0_heard = L0(ctx.lex)[ctx.utterance]
    l1_heard = L1(params["alpha"], ctx.lex)[ctx.utterance]
    heard = (1.0 - params["w_literal"]) * l1_heard + params["w_literal"] * l0_heard

    return jnp.where(ctx.is_prior > 0, prior, heard)
