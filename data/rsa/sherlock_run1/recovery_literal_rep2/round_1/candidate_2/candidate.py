"""RSA pragmatic listener with a contextual feature rarity prior.

Rather than treating all features as contributing equally to an object's
salience, the prior weights each feature inversely by its contextual
frequency (the number of objects sharing it in the display). An object with
unique, distinctive features is thus more salient a priori than an object
with generic, widely shared features. This prior enters as common knowledge
at both the literal listener and pragmatic listener levels.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_rarity": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_utt = (1.0 - ctx.is_sink)[:, None] * ctx.lex
    n_referents = jnp.sum(real_utt, axis=-1)
    rarity_weight = jnp.where(ctx.is_sink > 0, 0.0, 1.0 / (n_referents + EPS))
    rarity_score = jnp.sum(real_utt * rarity_weight[:, None], axis=0)

    prior = softmax_prior(
        params["w_rarity"] * rarity_score + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
