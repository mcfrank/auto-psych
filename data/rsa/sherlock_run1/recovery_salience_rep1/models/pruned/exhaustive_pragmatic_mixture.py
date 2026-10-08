"""Population mixture of grammatical exhaustification and pragmatic listeners.

Listeners across the population are heterogeneous: a proportion of listeners
are pragmatic (L1), inverting a model of a rational communicative speaker,
while the remaining proportion resolve reference via grammatical exhaustification
(L_exh), penalizing candidate objects in proportion to their unmentioned extraneous
features.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 0.5),
    "w_pragmatic": dist.Beta(1.0, 1.0),
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


@memo
def L_exh[u: UTT, r: OBJ](beta, lex: ..., extra_features: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * vec(extra_features, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    extra = jnp.maximum(ctx.feature_count - 1.0, 0.0)
    p_pragmatic = L1(params["alpha"], ctx.lex)[ctx.utterance]
    p_exh = L_exh(params["beta"], ctx.lex, extra)[ctx.utterance]
    heard = params["w_pragmatic"] * p_pragmatic + (1.0 - params["w_pragmatic"]) * p_exh
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
