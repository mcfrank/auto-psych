"""Evaluative listener reasoning about an affective, value-conditioned speaker.

The listener models a speaker who communicates about a referent chosen
according to their evaluative attitude (valence): a speaker describing their
favorite object favors referents with more features, while a speaker
describing their least favorite object favors referents with fewer features.
On neutral trials, referent preference is uniform. The listener inverts this
value-conditioned speaker upon hearing an utterance, and chooses by the
evaluative preference when no informative word is given. A lapse parameter
mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta_valence": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., eval_prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(eval_prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    eval_prior = softmax_prior(params["beta_valence"] * ctx.valence * ctx.feature_count)
    heard = L1(params["alpha"], ctx.lex, eval_prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, eval_prior, heard), params["lapse"])
