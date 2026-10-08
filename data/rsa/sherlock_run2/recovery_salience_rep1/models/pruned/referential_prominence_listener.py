"""Referential prominence listener: perceptual and evaluative framing in pragmatic reference.

Hypothesis: Listeners model a speaker whose referential choices are guided by
an object's contextual prominence. An object's prominence increases when it
stands out in visual color against grayscale competitors, while its feature
complexity is evaluated relative to the communicative frame—favoring simpler,
prototypical objects under neutral framing, but shifting toward multi-feature
objects under negative evaluative framing ("least favorite"). On prior trials,
the listener chooses directly according to this referential prominence prior.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "simplicity": dist.Normal(0.0, 1.0),
    "color_salience": dist.Normal(0.0, 1.0),
    "valence_weight": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    color_prominence = params["color_salience"] * (1.0 - ctx.grayscale)
    feature_weight = params["simplicity"] - params["valence_weight"] * ctx.valence
    feature_prominence = feature_weight * ctx.feature_count
    prior = softmax_prior(color_prominence + feature_prominence)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    probs = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(probs, params["lapse"])
