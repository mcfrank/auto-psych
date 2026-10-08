"""RSA pragmatic listener at depth 1 with softmax decision rationality.

Listeners infer communicative intentions by simulating a rational speaker at
depth 1, but do not merely probability-match their posterior beliefs when
clicking an object. Instead, they apply softmax decision rationality beta to
their posterior beliefs, separating the listener's choice decisiveness from
the speaker's communicative optimization.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L1(params["alpha"], params["beta"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
