"""Perceptual color-salience pragmatic listener.

Listeners interpret referring expressions under a perceptual salience prior
governed by visual color contrast: objects rendered in full color against a
grayscale background capture bottom-up visual attention, increasing the prior
probability that the speaker intends to refer to them. When no object stands
out in color (such as displays where all objects share the same presentation),
visual attention is uniform, while on displays with color contrast, the
highlighted object receives a perceptual prior advantage that guides both
unprompted choices and ambiguous reference resolution.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "color_weight": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, prior: ..., lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    color_signal = 1.0 - ctx.grayscale
    prior = softmax_prior(params["color_weight"] * color_signal)
    heard = L1(params["alpha"], prior, ctx.lex)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
