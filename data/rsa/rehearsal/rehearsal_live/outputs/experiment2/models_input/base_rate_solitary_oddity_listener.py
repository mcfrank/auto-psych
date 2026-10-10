"""Solitary perceptual oddity heuristic listener with familiarization base rates.

Refining solitary_oddity_heuristic_listener by incorporating contextual
familiarization base-rate weighting. Listeners interpret referring expressions
using a perceptual oddity heuristic rather than recursive Theory of Mind,
favoring solitary singletons that contrast against identical background duplicates,
while additionally weighting candidate referents by their observed familiarization
base rates. On uninformative prior trials without an informative word, listener
choices follow both solitary oddity pop-out and familiarization frequency; when an
informative referring expression is heard, listeners choose among semantically
matching referents weighted by both solitary oddity and familiarization frequency.
A lapse parameter captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 2.0),
    "w_base": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](beta, w_base, lex: ..., is_singleton: ..., base_rate: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * exp(beta * vec(is_singleton, r) + w_base * vec(base_rate, r)),
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
    is_solitary_singleton = jnp.where(
        (is_item_singleton > 0.0) & (singleton_count > 0.5) & (singleton_count < 1.5),
        1.0,
        0.0,
    )
    return is_solitary_singleton


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    base_rate = ctx.familiarization
    prior = softmax_prior(params["beta"] * is_singleton + params["w_base"] * base_rate)
    heard = L_heur(
        params["beta"], params["w_base"], ctx.lex, is_singleton, base_rate
    )[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
