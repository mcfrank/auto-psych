"""Discrete mixture of literal and pragmatic listener types in the population.

Rather than assuming all individuals execute an identical resource-bounded
tradeoff, this model proposes that the population of listeners is heterogeneous,
consisting of a discrete mixture of literal and pragmatic listeners. Literal
listeners select among truthful referents with a feature economy preference
penalizing extraneous unmentioned features. Pragmatic listeners recursively infer
the speaker's communicative intentions by simulating what an informative speaker
would say. Observed population choices reflect a convex mixture of these two
distinct cognitive types.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(-2.0, 1.0),
    "alpha": dist.Gamma(2.0, 1.0),
    "p_pragmatic": dist.Beta(2.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_literal[u: UTT, r: OBJ](beta, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * vec(feature_count, r)),
    )
    return Pr[listener.r == r]


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def S1[u: UTT, r: OBJ](alpha, lex: ...):
    speaker: given(r in OBJ, wpp=1)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L_pragmatic[u: UTT, r: OBJ](alpha, lex: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=L0[u, r](lex) * S1[u, r](alpha, lex),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    p_lit = L_literal(params["beta"], ctx.lex, ctx.feature_count)[ctx.utterance]
    p_prag = L_pragmatic(params["alpha"], ctx.lex)[ctx.utterance]
    p = params["p_pragmatic"]
    heard = (1.0 - p) * p_lit + p * p_prag
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
