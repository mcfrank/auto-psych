"""Fewest-features heuristic listener.

Hypothesis: Listeners do not reason counterfactually about an informative
speaker; instead, they follow a direct simplicity heuristic that favors objects
with fewer features. When hearing a word, listeners filter the visual display to
objects literally matching that word and select among them with a bias toward
objects with fewer total features. In the absence of an informative utterance,
listeners rely directly on this simplicity preference across all available
objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse, softmax_prior

PARAMS = {
    "simplicity": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., weights: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(weights, r))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    weights = softmax_prior(-params["simplicity"] * ctx.feature_count)
    heard = L0(ctx.lex, weights)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, weights, heard), params["lapse"])
