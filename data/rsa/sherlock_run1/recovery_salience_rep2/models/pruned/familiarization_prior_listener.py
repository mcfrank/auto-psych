"""Bayesian pragmatic listener with a familiarization base-rate prior.

The listener integrates the speaker's informativeness with an empirical prior
over referents established during familiarization (Frank & Goodman, 2012).
When no informative utterance is heard, choices follow this prior directly.
Displays without familiarization default to a uniform prior.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
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
    n_obj = ctx.familiarization.shape[0]
    uniform = jnp.full_like(ctx.familiarization, 1.0 / n_obj)
    prior = jnp.where(ctx.has_familiarization > 0, ctx.familiarization, uniform)
    prior = (prior + EPS) / jnp.sum(prior + EPS)

    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
