"""RSA pragmatic listener with graded truth semantics.

The semantic applicability of a feature word is graded rather than binary:
an object possessing only that feature is a prototypical match (weight 1),
while the word's applicability decays exponentially with each additional
extraneous feature on the object. Listeners invert a speaker whose word
production is guided by this graded semantic fit.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "decay": dist.HalfNormal(1.0),
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
    is_real = (1.0 - ctx.is_sink)[:, None]
    extraneous = jnp.maximum(0.0, ctx.feature_count[None, :] - 1.0) * is_real
    graded_lex = ctx.lex * jnp.exp(-params["decay"] * extraneous)
    heard = L1(params["alpha"], graded_lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
