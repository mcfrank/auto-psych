"""Twin-contrast solitary perceptual oddity listener.

Refining attenuated_solitary_oddity_listener by restricting the solitary oddity
heuristic to visual scenes where a solitary singleton contrasts against an
identical twin pair (maximum copy count equals 2). In triad displays with an
identical twin pair and one unique object, the solitary item pops out as an
odd-man-out against the duplicate pair. When a display contains a majority
triplet of identical items, the triplet forms a dominant visual category
rather than a twin contrast, and listeners distribute baseline expectations
without solitary oddity pop-out.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_prior": dist.Normal(0.0, 2.0),
    "beta_utt": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](beta_utt, lex: ..., is_singleton: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(beta_utt * vec(is_singleton, r)),
    )
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
    prior = softmax_prior(params["beta_prior"] * is_singleton)
    heard = L_heur(params["beta_utt"], ctx.lex, is_singleton)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
