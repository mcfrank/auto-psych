"""Evaluative proportional contrast: proportional feature contrast modulated by communicative framing.

Listeners interpret referring expressions through proportional feature contrast, evaluating
matching referents by penalizing unmentioned features relative to each referent's total feature
complexity, with the direction of contrast modulated by communicative framing. Under neutral or
positive framing, candidate referents are penalized for unmentioned features because speakers are
expected to provide distinguishing descriptions; under evaluative framing ('least favorite'),
unmentioned features represent compounding negative attributes that reverse this penalty to favor
more complex referents.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "theta": dist.LogNormal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_contrast[u: UTT, r: OBJ](coeff, lex: ..., unmentioned: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r) * exp(coeff * at(unmentioned, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(params["w_familiar"] * ctx.familiarization)
    unmentioned_count = jnp.maximum(0.0, ctx.feature_count[None, :] - ctx.lex)
    unmentioned = unmentioned_count / jnp.maximum(1.0, ctx.feature_count[None, :])
    coeff = -params["theta"] - params["w_valence"] * ctx.valence
    heard = L_contrast(coeff, ctx.lex, unmentioned, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
