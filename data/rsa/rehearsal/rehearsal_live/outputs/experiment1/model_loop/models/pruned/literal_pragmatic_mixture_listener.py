"""Literal-pragmatic population mixture listener.

Listeners differ in their depth of communicative reasoning: the population is
a mixture of literal listeners (who choose among referents matching literal
semantics) and pragmatic listeners (who invert an informative speaker). On
trials with an informative referring expression, aggregate choices reflect this
population mixture of literal and pragmatic listener strategies. On prior
trials with no informative word, choices default to uniform guessing. A lapse
parameter captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_prag": dist.Beta(2.0, 2.0),
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


def choice_probs(params, ctx):
    p_l0 = L0(ctx.lex)[ctx.utterance]
    p_l1 = L1(params["alpha"], ctx.lex)[ctx.utterance]
    heard = params["w_prag"] * p_l1 + (1.0 - params["w_prag"]) * p_l0
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
