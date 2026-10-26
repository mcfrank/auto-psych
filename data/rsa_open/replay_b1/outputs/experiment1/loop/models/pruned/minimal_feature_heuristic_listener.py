import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
}


@memo
def L_min[u: UTT, r: OBJ](beta, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * vec(feature_count, r)),
    )
    return Pr[listener.r == r]


def listener(params, ctx):
    prior = softmax_prior(-params["beta"] * ctx.feature_count)
    heard = L_min(params["beta"], ctx.lex, ctx.feature_count)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
