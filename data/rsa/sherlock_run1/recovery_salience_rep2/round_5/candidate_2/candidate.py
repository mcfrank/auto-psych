"""Lexical preemption cost speaker.

When an intended target possesses a uniquely distinguishing feature in the
context, that feature preempts shared, ambiguous alternatives, imposing a cost
on producing ambiguous words for that referent. When a target has no unique
feature, all true descriptions are evaluated without penalty. Pragmatic listeners
reason about this speaker at depth 2.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "preemption_cost": dist.HalfNormal(1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, preemption_cost, lex: ..., is_preempted: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * (
                    log(L1[u, r](alpha, lex) + {EPS})
                    - preemption_cost * at(is_preempted, u, r)
                )
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

    heard = L2(params["alpha"], params["preemption_cost"], ctx.lex, is_preempted)[
        ctx.utterance
    ]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
