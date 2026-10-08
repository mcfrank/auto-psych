"""Pragmatic listener combining feature salience with familiarization base rates.

Refines feature_salience_listener by integrating empirical familiarization base
rates into the listener's referent prior when available.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "salience_weight": dist.Normal(0.0, 1.0),
    "base_rate_weight": dist.Beta(2.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, prior: ..., lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    salience_prior = softmax_prior(params["salience_weight"] * ctx.feature_count)
    fam_prior = (1.0 - params["base_rate_weight"]) * salience_prior + params["base_rate_weight"] * ctx.familiarization
    prior = jnp.where(ctx.has_familiarization > 0, fam_prior, salience_prior)
    heard = L1(params["alpha"], prior, ctx.lex)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
