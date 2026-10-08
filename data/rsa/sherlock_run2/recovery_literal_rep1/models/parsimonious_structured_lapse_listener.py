"""Parsimonious listener with salience-structured attentional lapse.

Refinement of salience_structured_lapse_listener: listeners interpret utterances
by selecting among referents satisfying the heard word with a parsimony preference
for objects with fewer features. When goal-directed attention lapses or when no
informative clue is provided, choice falls back to bottom-up contextual salience.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., weight: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(weight, r))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    salience = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    weight = jnp.exp(-params["beta"] * ctx.feature_count)
    heard = L0(ctx.lex, weight)[ctx.utterance]
    return jnp.where(
        ctx.is_prior > 0,
        salience,
        (1.0 - params["lapse"]) * heard + params["lapse"] * salience,
    )
