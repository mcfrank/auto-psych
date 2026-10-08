"""Distractor attenuation listener with empirical base-rate prior.

Refinement of distractor_attenuation_listener: listeners operate under visual
attentional constraints that attenuate non-matching distractors during speaker
simulation, while additionally incorporating empirical familiarization base
rates into referent expectations (from base_rate_prior_l2) both during speaker
simulation and as the default choice on uninformative baseline trials.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "distractor_weight": dist.Beta(2.0, 2.0),
    "base_rate_weight": dist.Normal(0.0, 2.0),
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
    prior = softmax_prior(params["base_rate_weight"] * ctx.familiarization)
    heard = L1(params["alpha"], ctx.lex, atten, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
