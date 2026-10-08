"""Pragmatic listener with subjective probability weighting decision rule.

Listeners perform standard depth-1 pragmatic inference over shared salience priors,
but turn their posterior beliefs into choices via an inverted S-shaped probability
weighting function (Prelec 1998). This psychophysical decision rule overweights
improbable competitor referents and underweights highly probable target referents,
directly accounting for human conservatism in referent choice.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 0.5),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]

    # Prelec probability weighting function on posterior beliefs
    eps = 1e-6
    neg_log = -jnp.log(jnp.clip(heard, eps, 1.0 - eps))
    x = jnp.maximum(neg_log, 1e-4)
    w = jnp.exp(-(x ** params["gamma"]))
    distorted = w / jnp.sum(w, axis=-1, keepdims=True)

    choice = jnp.where(ctx.is_prior > 0, prior, distorted)
    return with_lapse(choice, params["lapse"])
