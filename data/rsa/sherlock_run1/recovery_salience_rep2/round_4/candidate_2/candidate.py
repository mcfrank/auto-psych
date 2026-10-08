"""Cognitive hierarchy mixture over listener reasoning depths.

Listeners in the population differ in their depth of recursive social reasoning:
a small proportion are non-strategic literal listeners (L0), while pragmatic
listeners are divided between first-order (L1) and second-order (L2) reasoners.
Aggregate choices reflect a population mixture over these discrete listener types.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "literal_weight": dist.Beta(1.0, 9.0),
    "depth2_weight": dist.Beta(2.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
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
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    p_l0 = L0(ctx.lex)[ctx.utterance]
    p_l1 = L1(params["alpha"], ctx.lex)[ctx.utterance]
    p_l2 = L2(params["alpha"], ctx.lex)[ctx.utterance]

    w_lit = params["literal_weight"]
    w_d2 = (1.0 - w_lit) * params["depth2_weight"]
    w_d1 = (1.0 - w_lit) * (1.0 - params["depth2_weight"])

    mixed = w_lit * p_l0 + w_d1 * p_l1 + w_d2 * p_l2
    uniform = jnp.full_like(mixed, 1.0 / mixed.shape[0])
    choice = jnp.where(ctx.is_prior > 0, uniform, mixed)
    return with_lapse(choice, params["lapse"])
