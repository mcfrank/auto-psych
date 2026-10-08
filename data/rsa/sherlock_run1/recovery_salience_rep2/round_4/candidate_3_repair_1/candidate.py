"""Valence-salience pragmatic listener.

Listeners interpret referring expressions under an evaluative salience prior
governed by affective valence: when a speaker refers to their favorite object
(positive valence), listeners expect them to favor visually richer referents
with more features, whereas when describing their least favorite object
(negative valence), this evaluative prior inverts to favor simpler referents.
On neutral trials where no evaluative attitude is expressed, expectations
reduce to an equiprobable baseline.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "valence_weight": dist.Normal(0.0, 1.0),
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
    evaluative_signal = ctx.valence * ctx.feature_count
    prior = softmax_prior(params["valence_weight"] * evaluative_signal)
    heard = L1(params["alpha"], prior, ctx.lex)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
