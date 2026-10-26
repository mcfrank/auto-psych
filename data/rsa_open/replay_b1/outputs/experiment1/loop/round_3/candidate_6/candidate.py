"""Contrast distinct empty listener.

Refines contextual_contrast_distinct_listener by incorporating empty-object baseline salience
into the perceptual referent prior alongside discrete singleton salience and continuous distinctiveness.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_empty": dist.Normal(0.0, 1.0),
}


@memo
def L_contrast[u: UTT, r: OBJ](alpha, lex: ..., contrast: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * at(contrast, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words

    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    dist = jnp.sum(diff, axis=0)

    n_obj = dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)

    # Contextual contrast of utterance u for referent r against distractors lacking u
    contrast = real_lex * ((1.0 - real_lex) @ sim)

    # Perceptual prior: discrete singleton salience, continuous distinctiveness, and empty baseline salience
    duplicate_count = jnp.sum(dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)
    distinct = jnp.sum(dist, axis=1)
    is_empty = jnp.where(ctx.feature_count == 0, 1.0, 0.0)

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_empty"] * is_empty
    )

    heard = L_contrast(params["alpha"], ctx.lex, contrast, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
