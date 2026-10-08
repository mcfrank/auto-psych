"""Contextual distinctiveness penalty listener.

Listeners resolve reference through a heuristic penalty on contextual distinctiveness
rather than recursive speaker mentalizing: hearing a feature word, they choose among
matching candidate referents while penalizing candidates whose extraneous features
are rare or unique across the visual scene. On uninformative prior trials, listeners
similarly favor simpler, less distinctive referents.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse, softmax_prior

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_dist[u: UTT, r: OBJ](beta, lex: ..., distinctiveness: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * vec(distinctiveness, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    feat_freq = real_lex.sum(axis=1)
    safe_freq = jnp.where(feat_freq > 0.0, feat_freq, 1.0)
    distinctiveness = (real_lex * (1.0 / safe_freq[:, None])).sum(axis=0)

    prior = softmax_prior(-params["beta"] * distinctiveness)
    heard = L_dist(params["beta"], ctx.lex, distinctiveness)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
