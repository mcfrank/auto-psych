"""Attenuated solitary perceptual oddity listener with twin contrast, evaluative valence, base rates, and feature complexity.

Refining twin_contrast_valence_base_rate_listener by incorporating visual feature complexity
weighting into the ungrounded prior over candidate referents from twin_contrast_feature_prior_listener.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_prior": dist.Normal(0.0, 2.0),
    "beta_utt": dist.Normal(0.0, 2.0),
    "w_valence": dist.Normal(0.0, 2.0),
    "w_base": dist.Normal(0.0, 2.0),
    "w_feat": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](
    beta_utt, w_base, lex: ..., is_singleton: ..., valence_bias: ..., base_rate: ...
):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * exp(
            beta_utt * vec(is_singleton, r)
            + vec(valence_bias, r)
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
    valence_bias = params["w_valence"] * ctx.valence * ctx.feature_count
    base_rate = ctx.familiarization
    prior = softmax_prior(
        params["beta_prior"] * is_singleton
        + valence_bias
        + params["w_base"] * base_rate
        + params["w_feat"] * ctx.feature_count
    )
    heard = L_heur(
        params["beta_utt"],
        params["w_base"],
        ctx.lex,
        is_singleton,
        valence_bias,
        base_rate,
    )[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
