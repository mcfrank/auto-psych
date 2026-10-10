"""Twin-contrast perceptual oddity heuristic listener.

Refining solitary_oddity_heuristic_listener by restricting the perceptual oddity
heuristic to singletons contrasting specifically against an identical twin pair.
Listeners interpret referring expressions using a perceptual oddity heuristic
rather than recursive Theory of Mind, but an oddity pop-out bias operates only
when exactly one unique singleton contrasts against an identical twin pair
(duplicate count of 2) in the visual context. When background duplicates form
a triplet majority (duplicate count of 3), the 3-to-1 category base rate cancels
perceptual pop-out, and listeners choose uniformly among semantically matching
referents. On uninformative prior trials without an informative word, listeners
spontaneously select the twin-contrasted singleton, defaulting to uniform choice
otherwise. A lapse parameter captures random clicking.
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
    is_item_singleton = (copy_count <= 1.0).astype(jnp.float32)
    singleton_count = jnp.sum(is_item_singleton)
    max_copy = jnp.max(copy_count)
    is_solitary_singleton = jnp.where(
        (is_item_singleton > 0.0)
        & (singleton_count > 0.5)
        & (singleton_count < 1.5)
        & (max_copy > 1.5)
        & (max_copy < 2.5),
        1.0,
        0.0,
    )
    return is_solitary_singleton


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    prior = softmax_prior(params["beta"] * is_singleton)
    heard = L_heur(params["beta"], ctx.lex, is_singleton)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
