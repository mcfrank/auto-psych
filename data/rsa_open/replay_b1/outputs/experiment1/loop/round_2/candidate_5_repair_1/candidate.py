"""Contextual contrast listener with visual feature complexity prior.

Refining contextual_contrast_listener by incorporating visual feature complexity from
rsa_l2_singleton_feat_color_valence_l0 into the referent prior: speakers choose referring expressions
to maximize contextual contrast against visually similar competitors in the immediate display, and
listeners interpret utterances by inverting this contrastive speaker. Both in the communicative speaker's
referent expectations and on uninformative prior displays, visual salience combines discrete visual singleton
pop-out (identifying unique objects with no duplicates) with visual feature complexity (favoring objects
possessing more visual features).
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
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

    # Perceptual prior: discrete singleton salience and visual feature complexity
    duplicate_count = jnp.sum(dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)
    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_features"] * ctx.feature_count
    )

    heard = L_contrast(params["alpha"], ctx.lex, contrast, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
