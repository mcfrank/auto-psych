"""Non-Bayesian heuristic listener with feature economy simplicity prior.

Refines fewest_features_listener by extending the feature economy heuristic to prior
expectations over referents: listeners possess an intrinsic cognitive preference for
minimal, less complex objects that operates both when interpreting utterances and
when forming a priori expectations. Rather than reverting to a uniform distribution
on prior trials without an informative utterance, choices favor simpler objects
with fewer total features according to the same feature economy penalty.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heuristic[u: UTT, r: OBJ](beta, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * vec(feature_count, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L_heuristic(params["beta"], ctx.lex, ctx.feature_count)[ctx.utterance]
    prior = softmax_prior(-params["beta"] * ctx.feature_count)
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
