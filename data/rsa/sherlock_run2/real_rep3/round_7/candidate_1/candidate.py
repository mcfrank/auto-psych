"""Distractor overlap heuristic listener.

Listeners interpret referring expressions using a non-Bayesian distractor-interference
heuristic rather than recursive Theory of Mind. When hearing a feature word,
listeners choose among candidate objects possessing that feature by penalizing
candidates that share unmentioned visual features with non-matching distractors
in the display, favoring referents that are perceptually distinct from the
non-target context. On uninformative trials without an informative word,
listeners choose uniformly among available objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](beta, lex: ..., overlap: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * exp(-beta * at(overlap, u, r)))
    return Pr[listener.r == r]


def compute_distractor_overlap(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    feats = ctx.lex * real_words
    omega = feats.T @ feats

    distractors = 1.0 - ctx.lex
    overlap = distractors @ omega

    n_distractors = jnp.sum(distractors, axis=1, keepdims=True)
    mean_overlap = overlap / jnp.maximum(1.0, n_distractors)
    return mean_overlap


def choice_probs(params, ctx):
    overlap = compute_distractor_overlap(ctx)
    heard = L_heur(params["beta"], ctx.lex, overlap)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
