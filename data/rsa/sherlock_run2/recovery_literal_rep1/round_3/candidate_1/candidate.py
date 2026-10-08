"""Framed specificity listener: descriptive specificity modulated by perceptual and affective framing."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, with_lapse

PARAMS = {
    "w_spec": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](affinity: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(affinity, u, r))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    excess = jnp.maximum(0.0, ctx.feature_count - 1.0)
    is_color = 1.0 - ctx.grayscale

    log_weight = (
        (-params["w_spec"] - params["w_valence"] * ctx.valence) * excess
        + params["w_color"] * is_color
        + params["w_familiar"] * ctx.familiarization
    )
    log_weight = jnp.clip(log_weight, -15.0, 15.0)
    affinity = ctx.lex * jnp.exp(log_weight)[None, :] + ctx.is_sink[:, None] * ctx.lex

    heard = L0(affinity)[ctx.utterance]

    prior_logits = (
        (params["w_features"] - params["w_valence"] * ctx.valence) * ctx.feature_count
        + params["w_color"] * is_color
        + params["w_familiar"] * ctx.familiarization
    )
    prior = softmax_prior(prior_logits)

    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
