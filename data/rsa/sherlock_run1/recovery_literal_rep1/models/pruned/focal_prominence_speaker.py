"""Pragmatic listener inverting a focal prominence-gated speaker.

Speakers tailor their specificity demand to referent prominence: for prominent
focal objects (highlighted by color contrast, feature complexity, or evaluative
framing), speakers readily use shared features, whereas for background objects
they demand discriminating specificity. The listener inverts this production
process, integrating linguistic evidence with a visual-affective prominence prior.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "lambda_spec": dist.HalfNormal(1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, utt_weights: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(utt_weights, u, r)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(alpha * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prominence = (
        (params["w_features"] + params["w_valence"] * ctx.valence) * ctx.feature_count
        + params["w_familiar"] * ctx.familiarization
        + params["w_color"] * (1.0 - ctx.grayscale)
    )
    prior = softmax_prior(prominence)

    # Feature specificity: 1 / extension across objects in the display
    ext = jnp.sum(ctx.lex * (1.0 - ctx.is_sink[:, None]), axis=1)
    spec = (1.0 - ctx.is_sink) / (ext + EPS) + ctx.is_sink

    # Specificity demand scales inversely with object prominence:
    # non-prominent objects require high specificity to overcome the background
    lam_r = params["lambda_spec"] * (1.0 - prior)

    # Production weight: lex[u, r] * spec[u] ** lam_r[r]
    log_spec = jnp.log(spec + EPS)[:, None] * lam_r[None, :]
    utt_weights = ctx.lex * jnp.exp(log_spec)

    heard = L1(params["alpha"], utt_weights, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
