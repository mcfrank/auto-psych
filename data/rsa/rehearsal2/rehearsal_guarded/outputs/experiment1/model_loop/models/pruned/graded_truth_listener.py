"""Pragmatic listener reasoning under graded truth semantics.

Listeners interpret referring expressions using graded truth values, where a
word's semantic applicability to an object diminishes as the object possesses
extraneous competing visual features. Rather than treating truth as an all-or-
nothing binary match, listeners perceive words as less prototypical descriptions
of cluttered multi-feature objects. Pragmatic listeners invert speakers who
communicate using these graded semantic applicability values at depth 2.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.HalfNormal(10.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](soft_lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(soft_lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, soft_lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(soft_lex, u, r) * exp(alpha * log(L0[u, r](soft_lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, soft_lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(soft_lex, u, r) * exp(alpha * log(L1[u, r](alpha, soft_lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    fc = jnp.maximum(ctx.feature_count, 1.0)
    extraneous_ratio = (fc - 1.0) / fc
    precision = jnp.maximum(jnp.exp(-params["gamma"] * extraneous_ratio), 1e-3)
    real_truth = ctx.lex * precision[None, :]
    soft_lex = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, real_truth)

    heard = L2(params["alpha"], soft_lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
