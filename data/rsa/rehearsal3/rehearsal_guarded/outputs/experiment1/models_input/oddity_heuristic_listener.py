"""Perceptual oddity heuristic listener.

Listeners interpret referring expressions using a perceptual oddity heuristic
rather than recursive Theory of Mind. When hearing a word that applies to
multiple referents, listeners select the contextually unique singleton that
lacks identical duplicates in the visual display. On uninformative trials
without an informative word, listeners spontaneously choose the unique
singleton, defaulting to uniform choice when all items are distinct or no
oddity contrast exists. A lapse parameter captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](beta, lex: ..., is_singleton: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * exp(beta * vec(is_singleton, r)))
    return Pr[listener.r == r]


def compute_singleton_indicator(ctx):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    is_identical = (dist < 1e-5).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_singleton = (copy_count <= 1.0).astype(jnp.float32)
    return is_singleton


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    prior = softmax_prior(params["beta"] * is_singleton)
    heard = L_heur(params["beta"], ctx.lex, is_singleton)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
