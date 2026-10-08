"""Literal descriptive specificity listener (depth 0).

Listeners interpret utterances literally according to contextual salience,
but weight candidate referents by their descriptive specificity: the proportion
of an object's features captured by the uttered word (the reciprocal of the
object's feature count). When a speaker names a single feature, objects
possessing that single feature are fully described (1/1 = 1.0), whereas
objects possessing multiple features are only partially specified (1/2 = 0.5),
naturally favoring minimally and precisely described referents. On prior trials
with no informative word, choice is guided solely by the salience prior.
A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "w_specificity": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](w_spec, lex: ..., prior: ..., spec: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r) * exp(w_spec * vec(spec, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    spec = 1.0 / jnp.maximum(1.0, ctx.feature_count)
    heard = L0(params["w_specificity"], ctx.lex, prior, spec)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
