"""Fewest features heuristic listener.

Listeners interpret referring expressions using a feature economy heuristic rather
than recursive Theory of Mind: when hearing an informative word, listeners choose
among semantically matching referents by preferring the object with the fewest
total visual features, penalizing extraneous unmentioned properties. On uninformative
prior trials without an informative word, listener choices track baseline visual
feature complexity.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_utt": dist.Normal(0.0, 2.0),
    "beta_prior": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](beta_utt, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta_utt * vec(feature_count, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(params["beta_prior"] * ctx.feature_count)
    heard = L_heur(params["beta_utt"], ctx.lex, ctx.feature_count)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
