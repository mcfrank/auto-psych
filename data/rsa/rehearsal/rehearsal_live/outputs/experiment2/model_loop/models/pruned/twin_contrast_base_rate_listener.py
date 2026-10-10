"""Twin-contrast solitary perceptual oddity listener with familiarization base rates.

Refining twin_contrast_solitary_oddity_listener by incorporating contextual
familiarization base-rate weighting from base_rate_solitary_oddity_listener.
Listeners interpret referring expressions using a solitary perceptual oddity
heuristic where an odd-one-out pop-out bias operates exclusively when a solitary
singleton contrasts against an identical twin pair (maximum copy count equals 2),
while additionally weighting candidate referents by their observed familiarization
base rates from prior exposure. On uninformative prior trials without an informative
word, choices follow both twin-contrast solitary oddity pop-out and familiarization
frequency; when an informative referring expression is heard, listeners choose among
semantically matching referents weighted by both twin-contrast oddity and familiarization
frequency. A lapse parameter captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_prior": dist.Normal(0.0, 2.0),
    "beta_utt": dist.Normal(0.0, 2.0),
    "w_base": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](
    beta_utt, w_base, lex: ..., is_singleton: ..., base_rate: ...
):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * exp(
            beta_utt * vec(is_singleton, r)
            + w_base * vec(base_rate, r)
        ),
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
    base_rate = ctx.familiarization
    prior = softmax_prior(
        params["beta_prior"] * is_singleton
        + params["w_base"] * base_rate
    )
    heard = L_heur(
        params["beta_utt"],
        params["w_base"],
        ctx.lex,
        is_singleton,
        base_rate,
    )[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
