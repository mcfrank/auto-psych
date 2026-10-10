"""Solitary perceptual oddity heuristic listener with mutual exclusivity.

Refining solitary_oddity_heuristic_listener by incorporating the mutual exclusivity
feature penalty from mutual_exclusivity_listener. Listeners interpret referring expressions
using a perceptual oddity heuristic that favors solitary visual singletons contrasting
against identical background duplicates. When referents are distinct, listeners resolve
ambiguity through the mutual exclusivity principle, penalizing any candidate referent
possessing an exclusive contextually unique feature that could have been used to identify it.
On uninformative prior trials, choices follow solitary oddity pop-out. A lapse parameter
captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_oddity": dist.Normal(0.0, 2.0),
    "beta_me": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](
    beta_oddity, beta_me, lex: ..., is_singleton: ..., exclusive_count: ...
):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * exp(
            beta_oddity * vec(is_singleton, r)
            - beta_me * vec(exclusive_count, r)
        ),
    )
    return Pr[listener.r == r]


def compute_singleton_indicator(ctx):
    feats = jnp.vstack([
        ctx.lex,
        ctx.grayscale[None, :],
        ctx.familiarization[None, :],
    ])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    is_identical = (dist < 1e-5).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_item_singleton = (copy_count <= 1.0).astype(jnp.float32)
    singleton_count = jnp.sum(is_item_singleton)
    is_solitary_singleton = jnp.where(
        (is_item_singleton > 0.0)
        & (singleton_count > 0.5)
        & (singleton_count < 1.5),
        1.0,
        0.0,
    )
    return is_solitary_singleton


def compute_exclusive_count(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    ext = jnp.sum(real_lex, axis=1, keepdims=True)
    is_exclusive_feat = (ext == 1.0).astype(jnp.float32) * real_words
    return jnp.sum(real_lex * is_exclusive_feat, axis=0)


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    exclusive_count = compute_exclusive_count(ctx)
    prior = softmax_prior(params["beta_oddity"] * is_singleton)
    heard = L_heur(
        params["beta_oddity"],
        params["beta_me"],
        ctx.lex,
        is_singleton,
        exclusive_count,
    )[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
