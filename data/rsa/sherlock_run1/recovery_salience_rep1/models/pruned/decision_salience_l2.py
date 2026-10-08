"""RSA pragmatic listener at depth 2 with feature salience and decision rationality.

Refinement of feature_salience_l2: listeners reason at depth 2 with a feature-count
salience prior, but apply softmax decision rationality beta when selecting a referent,
rather than strictly probability matching.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "w_feature": dist.Normal(0.0, 1.0),
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


@memo
def L2[u: UTT, r: OBJ](alpha, beta, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(params["w_feature"] * ctx.feature_count)
    heard = L2(params["alpha"], params["beta"], ctx.lex, prior)[ctx.utterance]
    predicted = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(predicted, params["lapse"])
