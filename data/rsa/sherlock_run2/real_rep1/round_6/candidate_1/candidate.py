"""Sample-budgeted rejection listener.

Listeners interpret referring expressions through sample-based mental simulation:
they evaluate candidate referents by checking whether a communicative speaker would
produce the heard description, but operate under a bounded budget of simulation samples.
If all simulation attempts fail because the heard word is an improbable or unnatural
description in the context, the resource-limited listener gives up and guesses uniformly
among the objects. Consequently, the effective rate of guessing is display- and
utterance-dependent, increasing when hearing ambiguous or unexpected words.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "k_samples": dist.LogNormal(1.0, 1.0),
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
def S1[u: UTT, r: OBJ](alpha, lex: ...):
    speaker: given(r in OBJ, wpp=1)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
    )
    return Pr[speaker.u == u]


def choice_probs(params, ctx):
    heard = L1(params["alpha"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])

    s_marginal = S1(params["alpha"], ctx.lex)[ctx.utterance, 0]
    p_hit = jnp.clip(s_marginal, 0.0, 1.0)
    p_miss = jnp.clip(1.0 - p_hit, EPS, 1.0)
    p_fail = jnp.exp(params["k_samples"] * jnp.log(p_miss))

    sample_probs = (1.0 - p_fail) * heard + p_fail * uniform
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, sample_probs), params["lapse"]
    )
