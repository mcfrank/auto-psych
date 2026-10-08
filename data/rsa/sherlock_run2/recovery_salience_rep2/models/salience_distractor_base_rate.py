"""Distractor attenuation listener with joint base-rate and feature-complexity prior.

Refinement of distractor_attenuation_base_rate: listeners operate under visual
attentional constraints that attenuate non-matching distractors during speaker
simulation, while evaluating candidate referents through an inductive prior
that combines empirical familiarization base rates with cognitive parsimony
(favoring objects with fewer visual features).
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "distractor_weight": dist.Beta(2.0, 2.0),
    "base_rate_weight": dist.Normal(0.0, 2.0),
    "salience": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., atten: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(atten, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., atten: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex, atten) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    matches = ctx.lex[ctx.utterance]
    atten = jnp.where(
        ctx.is_prior > 0,
        1.0,
        jnp.where(matches > 0, 1.0, params["distractor_weight"]),
    )
    prior = softmax_prior(
        params["base_rate_weight"] * ctx.familiarization
        + params["salience"] * ctx.feature_count
    )
    heard = L1(params["alpha"], ctx.lex, atten, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
