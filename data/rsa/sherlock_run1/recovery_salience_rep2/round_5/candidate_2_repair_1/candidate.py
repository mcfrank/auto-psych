"""Lexical preemption and unnameability speaker.

Speakers face lexical competition when choosing how to describe a referent:
an ambiguous, shared feature incurs a preemption penalty if a uniquely
distinguishing descriptor is available for that referent in context.
Conversely, when no informative word is produced (uninformative prior
trials), listeners infer that the speaker struggled to name the object,
biasing choices toward referents with fewer communicative features.
Pragmatic listeners invert this speaker at depth 1.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "preemption_cost": dist.HalfNormal(1.0),
    "unnameability_weight": dist.HalfNormal(1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, preemption_cost, lex: ..., is_preempted: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - preemption_cost * at(is_preempted, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    extension = jnp.sum(real_lex, axis=1)
    is_unique_utt = jnp.where(extension == 1.0, 1.0, 0.0) * (1.0 - ctx.is_sink)
    has_unique_utt = jnp.matmul(real_lex.T, is_unique_utt)
    obj_has_unique = jnp.where(has_unique_utt > 0, 1.0, 0.0)

    is_preempted = obj_has_unique[None, :] * (1.0 - is_unique_utt[:, None]) * real_lex

    heard = L1(params["alpha"], params["preemption_cost"], ctx.lex, is_preempted)[
        ctx.utterance
    ]

    unnameability = softmax_prior(-params["unnameability_weight"] * ctx.feature_count)

    return with_lapse(jnp.where(ctx.is_prior > 0, unnameability, heard), params["lapse"])
