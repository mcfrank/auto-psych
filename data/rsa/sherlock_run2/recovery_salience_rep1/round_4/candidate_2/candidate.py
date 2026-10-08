"""Population mixture of literal and pragmatic listener types.

Hypothesis: The participant population consists of a mixture of distinct
listener types: literal listeners who choose uniformly among referents that
literally match the utterance, and pragmatic listeners who invert an
informative speaker to derive pragmatic implicatures. Observed choices reflect
the population mixture over these types.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "pragmatic_share": dist.Beta(1.0, 1.0),
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
    literal = L0(ctx.lex)[ctx.utterance]
    pragmatic = L1(params["alpha"], ctx.lex)[ctx.utterance]
    mixture = params["pragmatic_share"] * pragmatic + (1.0 - params["pragmatic_share"]) * literal
    uniform = jnp.full_like(mixture, 1.0 / mixture.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, mixture), params["lapse"])
