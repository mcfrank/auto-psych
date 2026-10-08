"""Residual surprisal listener: referent choice penalizing uncommunicated feature information."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., weight: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * at(weight, u, r))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(params["w_familiar"] * ctx.familiarization)

    is_real = 1.0 - ctx.is_sink
    real_lex = ctx.lex * is_real[:, None]
    ext = jnp.sum(real_lex, axis=1)
    n_obj = ctx.lex.shape[1]

    surprisal = jnp.log(n_obj / jnp.maximum(1.0, ext)) * is_real
    total_obj_surprisal = jnp.sum(real_lex * surprisal[:, None], axis=0)
    residual = jnp.maximum(0.0, total_obj_surprisal[None, :] - (surprisal[:, None] * real_lex))

    weight = jnp.exp(-params["beta"] * residual)
    heard = L0(ctx.lex, weight)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
