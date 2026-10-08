"""Power-law probability matching listener.

Listeners do not simply probability-match their subjective beliefs when
selecting a referent; instead, they convert posterior beliefs into choices via
power-law probability matching (the generalized matching law). A decision-precision
exponent governs how sharply beliefs translate into action, allowing listeners to
over-match by disproportionately favoring the most probable candidate referent.
"""

import jax
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
    beliefs = jnp.where(ctx.is_prior > 0, uniform, heard)
    logits = jnp.where(beliefs > 0, params["gamma"] * jnp.log(beliefs + EPS), -1e9)
    matched = jax.nn.softmax(logits)
    return with_lapse(matched, params["lapse"])
