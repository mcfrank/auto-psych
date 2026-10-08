"""Pragmatic listener reasoning about a speaker addressing a feature Question Under Discussion (QUD).

Rather than assuming the speaker's communicative goal is to identify the object's unique
identity, the listener models the speaker as answering a question under discussion about
an aspect of the object: whether it possesses a particular visual feature. The speaker
chooses an utterance to be informative about that feature question, and the pragmatic
listener jointly infers the speaker's question under discussion and intended referent.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.5, 0.5),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., p1: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(q in UTT, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * log(
                    0.99
                    * (
                        1.0
                        - at(p1, u, q)
                        + at(lex, q, r) * (2.0 * at(p1, u, q) - 1.0)
                    )
                    + 0.005
                    + {EPS}
                )
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    l0_table = L0(ctx.lex)
    p1 = jnp.dot(l0_table, ctx.lex.T)
    heard = L1(params["alpha"], ctx.lex, p1)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
