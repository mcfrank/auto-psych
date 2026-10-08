"""Pragmatic listener reasoning about a speaker penalized for uninformative alternatives.

Under Gricean Quantity, a speaker who uses an ambiguous word when a uniquely
distinguishing word for their intended referent was available in the visual
context incurs a communicative penalty. The pragmatic listener inverts this
speaker, recognizing that using an ambiguous word strongly indicates the
speaker lacked a distinguishing alternative.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, cost, lex: ..., cost_mat: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS}) - cost * at(cost_mat, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    extension = jnp.sum(real_lex, axis=1)
    is_unique = jnp.where(extension == 1.0, 1.0, 0.0)
    has_unique = jnp.where(jnp.sum(real_lex * is_unique[:, None], axis=0) > 0.0, 1.0, 0.0)
    is_ambiguous = jnp.where(extension > 1.0, 1.0, 0.0)
    cost_mat = is_ambiguous[:, None] * has_unique[None, :]

    heard = L1(params["alpha"], params["cost"], ctx.lex, cost_mat)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
