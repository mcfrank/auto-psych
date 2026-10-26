"""Contextual isolation listener.

Listeners interpret referring expressions by inverting a contrastive speaker while expecting
speakers to refer to contextually isolated outliers rather than items embedded in dense visual
clusters. An object's prior referential salience decreases with its contextual similarity density
to other items in the display, favoring distinctive singletons over duplicates and isolated items
over near-competitors. On uninformative displays without an informative word, listener referent
choices follow this contextual isolation prior directly.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_isolation": dist.Normal(0.0, 1.0),
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

    # Contextual isolation prior: negative similarity density to other objects
    density = jnp.sum(sim, axis=1)
    isolation = -density
    prior = softmax_prior(params["w_isolation"] * isolation)

    heard = L_contrast(params["alpha"], ctx.lex, contrast, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
