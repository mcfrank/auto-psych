"""Salience-structured lapse listener: literal semantic choice with salience-guided attentional fallback."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior

PARAMS = {
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    salience = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    heard = L0(ctx.lex)[ctx.utterance]
    return jnp.where(
        ctx.is_prior > 0,
        salience,
        (1.0 - params["lapse"]) * heard + params["lapse"] * salience,
    )
