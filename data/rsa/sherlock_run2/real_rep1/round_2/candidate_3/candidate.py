"""Pragmatic listener reasoning about a speaker communicating under lexical feature uncertainty.

Listeners interpret referring expressions under the hypothesis that word-feature mappings
are uncertain. The literal listener assumes an uttered word most likely refers to its
nominal feature but retains a non-zero probability of picking out an alternative feature
present in the context. A rational speaker anticipates this listener uncertainty when
choosing utterances, and the depth-1 pragmatic listener inverts that speaker.
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
def L0[u: UTT, r: OBJ](beta, lex: ...):
    listener: knows(u)
    listener: chooses(f in UTT, wpp=exp(beta * (f == u)))
    listener: chooses(r in OBJ, wpp=at(lex, f, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](beta, lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L1(params["alpha"], params["beta"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
