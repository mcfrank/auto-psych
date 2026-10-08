"""Pragmatic listener reasoning with a singleton salience prior.

Listeners expect speakers to refer to visual singletons: objects that possess
unique, unshared features in the visual scene. Before hearing an informative word
(or when evaluating choices on prior trials), listeners assign objects prior
expectation proportional to their count of contextually unique features.
Pragmatic listeners invert the speaker using this singleton salience prior.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_unique": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    col_sums = (ctx.lex * (1.0 - ctx.is_sink[:, None])).sum(axis=1, keepdims=True)
    is_unique = jnp.where(col_sums == 1.0, 1.0, 0.0)
    unique_count = (ctx.lex * (1.0 - ctx.is_sink[:, None]) * is_unique).sum(axis=0)
    prior = softmax_prior(params["w_unique"] * unique_count)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
