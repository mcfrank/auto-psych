"""Attenuated solitary perceptual oddity listener with twin contrast, attenuated color contrast, and attenuated valence framing.

Refining twin_color_attenuated_valence_listener by decoupling visual color contrast into separate
ungrounded prior (w_color_prior) and linguistic utterance comprehension (w_color_utt) weights.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_prior": dist.Normal(0.0, 2.0),
    "beta_utt": dist.Normal(0.0, 2.0),
    "w_valence_prior": dist.Normal(0.0, 2.0),
    "w_valence_utt": dist.Normal(0.0, 2.0),
    "w_color_prior": dist.Normal(0.0, 2.0),
    "w_color_utt": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](
    beta_utt, lex: ..., is_singleton: ..., valence_bias: ..., color_bias: ...
):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * exp(
            beta_utt * vec(is_singleton, r)
            + vec(valence_bias, r)
            + vec(color_bias, r)
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
    val_prior_bias = params["w_valence_prior"] * ctx.valence * ctx.feature_count
    val_utt_bias = params["w_valence_utt"] * ctx.valence * ctx.feature_count
    color_prior_bias = params["w_color_prior"] * (1.0 - ctx.grayscale)
    color_utt_bias = params["w_color_utt"] * (1.0 - ctx.grayscale)
    prior = softmax_prior(
        params["beta_prior"] * is_singleton + val_prior_bias + color_prior_bias
    )
    heard = L_heur(
        params["beta_utt"], ctx.lex, is_singleton, val_utt_bias, color_utt_bias
    )[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
