"""Feature contrast listener with distinctiveness-weighted unmentioned features.

Refines the incumbent feature_contrast_neutral_prior by incorporating contextual
distinctiveness (taken from contextual_distinctiveness_listener) into the evaluation
of unmentioned features. Listeners evaluate matching referents by penalizing unmentioned
features in proportion to their contextual distinctiveness (inverse frequency across
objects in the display), rather than penalizing all unmentioned features uniformly.
Rare distinguishing features that were omitted incur a strong contrast penalty, whereas
ubiquitous features shared across the visual display incur little penalty.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "theta": dist.LogNormal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_contrast[u: UTT, r: OBJ](theta, lex: ..., unmentioned: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r) * exp(-theta * at(unmentioned, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(params["w_familiar"] * ctx.familiarization)
    features = (1.0 - ctx.is_sink)[:, None] * ctx.lex
    freq = features.sum(axis=1)
    safe_freq = jnp.where(freq > 0.0, freq, 1.0)
    distinctiveness = (1.0 - ctx.is_sink) / safe_freq
    weighted_features = features * distinctiveness[:, None]
    total_weighted = weighted_features.sum(axis=0, keepdims=True)
    unmentioned = jnp.maximum(0.0, total_weighted - weighted_features)
    heard = L_contrast(params["theta"], ctx.lex, unmentioned, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
