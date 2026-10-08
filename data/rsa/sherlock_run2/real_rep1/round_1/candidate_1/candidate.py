"""Grammatical exhaustivity listener.

Listeners interpret referring expressions through grammatical exhaustification:
when hearing a feature word, the listener considers all candidate objects
possessing that feature and applies an exhaustification penalty for each
additional unmentioned feature true of the object in the context. On prior
trials with no informative word, choice is uniform. A lapse parameter mixes
in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse

PARAMS = {
    "lambda_exh": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_exh[u: UTT, r: OBJ](lambda_exh, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-lambda_exh * (vec(feature_count, r) - at(lex, u, r))),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L_exh(params["lambda_exh"], ctx.lex, ctx.feature_count)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
