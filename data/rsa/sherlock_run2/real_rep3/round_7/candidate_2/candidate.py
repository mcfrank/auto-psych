"""Margin-sensitive decision listener: listeners scale choice sharpness by the evidence margin.

In standard RSA, listeners are assumed to probability-match their pragmatic
posterior beliefs or apply a fixed-temperature softmax. Under this hypothesis,
listeners compute pragmatic posterior beliefs via recursive Theory of Mind, but
convert these beliefs into choices using a margin-sensitive decision rule. When
one referent clearly dominates the alternatives (a large margin between the top
and runner-up posterior probabilities), high decision confidence sharpens choice
toward probability maximizing. When evidence is conflicting or ambiguous (a small
margin), decision conflict flattens the choice rule toward probability matching.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 1.0),
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
    heard = L1(params["alpha"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    p = jnp.where(ctx.is_prior > 0, uniform, heard)

    # Margin-sensitive decision rule:
    # Scale choice sharpness by the probability margin between the best and runner-up options.
    sorted_p = jnp.sort(p)
    margin = sorted_p[-1] - sorted_p[-2]
    beta = 1.0 + params["gamma"] * margin
    weights = jnp.exp(beta * jnp.log(p + EPS))
    p_dec = weights / jnp.sum(weights)

    return with_lapse(p_dec, params["lapse"])
