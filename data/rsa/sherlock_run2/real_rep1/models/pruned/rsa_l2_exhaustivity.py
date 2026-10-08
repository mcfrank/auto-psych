"""RSA pragmatic listener at depth 2 with grammatical exhaustivity.

Refining rsa_l2 with the exhaustivity component from exhaustivity_listener:
listeners reason at depth 2 (inverting a speaker who simulates a depth-1
pragmatic listener), but the depth-2 pragmatic listener applies an
exhaustification penalty for each additional unmentioned feature an object
possesses in the context. This penalizes referents with superfluous competing
features. With no informative word (prior trials), the listener guesses
uniformly. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "lambda_exh": dist.LogNormal(0.0, 1.0),
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
def L2[u: UTT, r: OBJ](alpha, lambda_exh, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(
        r in OBJ,
        wpp=Pr[speaker.r == r]
        * exp(-lambda_exh * (vec(feature_count, r) - at(lex, u, r))),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L2(params["alpha"], params["lambda_exh"], ctx.lex, ctx.feature_count)[
        ctx.utterance
    ]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
