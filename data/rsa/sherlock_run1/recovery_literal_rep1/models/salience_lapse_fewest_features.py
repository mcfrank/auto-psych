"""Parsimony heuristic listener with salience-guided attentional lapse.

Refines fewest_features_listener: listeners choose literally matching referents
penalizing objects with extraneous features, but when attention lapses, choices
default to bottom-up perceptual salience rather than uniform random guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def HeuristicListener[u: UTT, r: OBJ](lex: ..., weights: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(weights, r))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    weights = jnp.exp(
        -params["beta"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    heard = HeuristicListener(ctx.lex, weights)[ctx.utterance]
    choice = (1.0 - params["lapse"]) * heard + params["lapse"] * prior
    return jnp.where(ctx.is_prior > 0, prior, choice)
