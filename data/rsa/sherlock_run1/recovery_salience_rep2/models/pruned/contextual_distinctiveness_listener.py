"""Contextual distinctiveness pragmatic listener.

Listeners combine pragmatic speaker informativeness with a contextual
distinctiveness prior over referents. Rather than assuming all objects
are equally likely a priori or scaling baseline expectations purely by
raw feature count, listeners form prior expectations governed by how
much an object contrasts with alternative items in the context (its average
feature Hamming distance to other objects in the display).
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "distinctiveness_weight": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, prior: ..., lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    obj_features = real_lex.T
    n_obj = obj_features.shape[0]
    diff = jnp.abs(obj_features[:, None, :] - obj_features[None, :, :])
    dist_matrix = jnp.sum(diff, axis=-1)
    mean_dist = jnp.sum(dist_matrix, axis=1) / jnp.maximum(1.0, n_obj - 1.0)

    prior = softmax_prior(params["distinctiveness_weight"] * mean_dist)
    heard = L1(params["alpha"], prior, ctx.lex)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
