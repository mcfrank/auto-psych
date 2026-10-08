"""Pragmatic listener inverting a speaker with a distinctiveness salience prior.

Rather than assuming prior salience is driven by raw feature count, listeners
expect speakers to refer to contextually distinctive objects—an isolation
effect where features that are rare or unique in the display capture attention.
Each feature's contribution to an object's prior salience is inversely
proportional to its extension across objects in the visual scene. The listener
inverts a descriptive speaker using this distinctiveness prior and decision
rationality alpha.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(alpha * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    ext = jnp.sum(ctx.lex * (1.0 - ctx.is_sink[:, None]), axis=1)
    spec = (1.0 - ctx.is_sink) / (ext + EPS)
    distinct = jnp.sum(ctx.lex * spec[:, None], axis=0)
    prior = softmax_prior(
        params["w_distinct"] * distinct + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
