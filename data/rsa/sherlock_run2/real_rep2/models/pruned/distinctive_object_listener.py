"""Pragmatic listener with contextual distinctiveness (visual pop-out) salience.

Listeners assume speakers refer to objects that visually pop out from the display.
An object's distinctiveness is measured as its mean Hamming distance to all other
objects in the context. The listener's prior over referents is a softmax of this
distinctiveness, weighting inference toward contrastive singleton objects and away
from duplicate items. With no informative word, choice follows this prior. A lapse
parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def object_distinctiveness(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Mean Hamming distance of each object to all other objects in the display."""
    features = lex * (1.0 - is_sink)[:, None]
    diff = jnp.abs(features[:, :, None] - features[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    n_obj = lex.shape[1]
    denom = jnp.maximum(n_obj - 1.0, 1.0)
    return jnp.sum(pair_dist, axis=1) / denom


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    dist_vec = object_distinctiveness(ctx.lex, ctx.is_sink)
    prior = softmax_prior(params["w_distinct"] * dist_vec)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
