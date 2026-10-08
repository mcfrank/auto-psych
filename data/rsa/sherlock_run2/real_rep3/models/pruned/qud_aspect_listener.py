"""Pragmatic listener reasoning about a speaker with feature-aspect communicative goals.

Rather than intending to uniquely identify an object token, the speaker selects
a communicative goal or Question Under Discussion (QUD) targeting an aspect
(feature) of the referent, choosing an utterance to maximize the literal listener's
recovery of that aspect. The pragmatic listener inverts this speaker, jointly
inferring the intended referent and the speaker's latent communicative goal.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0_aspect[u: UTT, q: UTT](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    listener: chooses(q_guess in UTT, wpp=at(lex, q_guess, r))
    return Pr[listener.q_guess == q]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            q in UTT,
            u in UTT,
            wpp=at(lex, q, r)
            * at(lex, u, r)
            * exp(alpha * log(L0_aspect[u, q](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L1(params["alpha"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
