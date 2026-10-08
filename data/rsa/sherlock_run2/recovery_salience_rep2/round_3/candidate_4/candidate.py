"""RSA pragmatic listener at depth 2 with base-rate prior, speaker ambiguity cost,
and distractor attenuation.

Refinement of base_rate_ambiguity_l2: incorporates distractor attenuation from
distractor_attenuation_listener. Listeners operate under visual attentional constraints,
allocating primary attention to referents matching the heard utterance while discounting
non-matching distractors during recursive speaker simulation.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost": dist.LogNormal(-1.0, 1.0),
    "base_rate_weight": dist.Normal(0.0, 2.0),
    "distractor_weight": dist.Beta(2.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., atten: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(atten, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, costs: ..., lex: ..., prior: ..., atten: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex, atten) + {EPS}) - vec(costs, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, costs: ..., lex: ..., prior: ..., atten: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, costs, lex, prior, atten) + {EPS})
                - vec(costs, u)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    extension = jnp.sum(ctx.lex * (1.0 - ctx.is_sink)[:, None], axis=1)
    costs = params["cost"] * jnp.maximum(0.0, extension - 1.0)
    prior = softmax_prior(params["base_rate_weight"] * ctx.familiarization)
    matches = ctx.lex[ctx.utterance]
    atten = jnp.where(
        ctx.is_prior > 0,
        1.0,
        jnp.where(matches > 0, 1.0, params["distractor_weight"]),
    )
    heard = L2(params["alpha"], costs, ctx.lex, prior, atten)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
