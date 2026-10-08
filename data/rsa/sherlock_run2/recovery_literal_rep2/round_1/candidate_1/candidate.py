"""Feature contrast listener: non-recursive pragmatic interpretation via unmentioned feature penalty.

Rather than inverting a counterfactual speaker, the listener evaluates candidate
referents directly according to feature contrast. Matching referents are penalized
for each additional unmentioned feature they possess, favouring objects whose
features are most exhaustively described. Prior-elicitation trials reflect the
object salience prior over feature count and familiarization base rate. A lapse
parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "theta": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_contrast[u: UTT, r: OBJ](theta, lex: ..., unmentioned: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r) * exp(-theta * at(unmentioned, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    unmentioned = jnp.maximum(0.0, ctx.feature_count[None, :] - ctx.lex)
    heard = L_contrast(params["theta"], ctx.lex, unmentioned, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
