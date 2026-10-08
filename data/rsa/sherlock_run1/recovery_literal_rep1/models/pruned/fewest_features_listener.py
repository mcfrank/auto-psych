"""Non-Bayesian heuristic listener selecting matching referents with fewest features.

Listeners interpret referring expressions using a non-Bayesian parsimony
heuristic rather than recursive speaker mentalization: upon hearing a word,
they identify candidate objects of which the word is literally true and
choose the candidate possessing the fewest total features.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

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
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
