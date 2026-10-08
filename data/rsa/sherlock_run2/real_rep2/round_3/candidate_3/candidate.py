"""Pragmatic listener with a softmax decision rule over posterior beliefs.

Rather than matching probabilities (the standard RSA assumption where click
probability directly equals the posterior belief), the listener applies a
softmax decision rule over posterior beliefs with choice determinism beta.
When beta > 1, the listener sharpens their preference toward the most probable
referent, turning subtle pragmatic advantages into decisive categorical clicks.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
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
    safe_heard = jnp.where(heard > 0, heard, 1.0)
    logits = params["beta"] * jnp.log(safe_heard)
    logits = logits - jnp.max(logits)
    weights = jnp.where(heard > 0, jnp.exp(logits), 0.0)
    choice = weights / (jnp.sum(weights) + EPS)
    uniform = jnp.full_like(choice, 1.0 / choice.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, choice), params["lapse"])
