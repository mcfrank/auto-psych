"""Mutual exclusivity heuristic listener.

Listeners interpret referring expressions using a non-Bayesian mutual exclusivity
heuristic rather than recursive Theory of Mind. When hearing a word that applies
to multiple candidate objects, listeners penalize candidates that possess
alternative distinguishing names in the visual scene, assuming each object has a
unique primary descriptor. On uninformative prior trials without an informative
word, listeners choose the object with the greatest communicative naming affordance.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 1.0),
    "beta_prior": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_me[u: UTT, r: OBJ](beta, lex: ..., alt_score: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * at(alt_score, u, r)),
    )
    return Pr[listener.r == r]


def compute_scores(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    k_u = jnp.sum(real_lex, axis=1)
    spec = (1.0 - ctx.is_sink) / jnp.maximum(1.0, k_u)
    total_namability = jnp.sum(real_lex * spec[:, None], axis=0)
    alt_score = total_namability[None, :] - real_lex * spec[:, None]
    return alt_score, total_namability


def choice_probs(params, ctx):
    alt_score, total_namability = compute_scores(ctx)
    prior = softmax_prior(params["beta_prior"] * total_namability)
    heard = L_me(params["beta"], ctx.lex, alt_score)[ctx.utterance]
    probs = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(probs, params["lapse"])
