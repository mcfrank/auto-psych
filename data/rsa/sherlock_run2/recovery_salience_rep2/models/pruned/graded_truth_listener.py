"""Graded truth listener.

Listeners evaluate referential expressions through graded semantic truth values,
where an utterance's degree of truth decreases with the referent's feature
complexity because additional attributes dilute the description's semantic fit.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](soft_lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(soft_lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, soft_lex: ..., lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](soft_lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    extra_features = jnp.maximum(0.0, ctx.feature_count - 1.0)
    truth_scale = jnp.exp(-params["gamma"] * extra_features)
    graded_lex = ctx.lex * truth_scale[None, :]
    soft_lex = jnp.where(ctx.is_sink[:, None] > 0.0, ctx.lex, graded_lex)

    prior = softmax_prior(-params["gamma"] * ctx.feature_count)
    heard = L1(params["alpha"], soft_lex, ctx.lex)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
