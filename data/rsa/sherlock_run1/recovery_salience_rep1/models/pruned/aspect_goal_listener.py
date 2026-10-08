"""Pragmatic listener reasoning about an aspect-oriented speaker with communicative goals.

Rather than assuming the speaker communicates to uniquely identify the referent's
complete identity, the listener models a speaker who selects an utterance to informatively
convey a specific aspect (feature) of the referent. The listener jointly infers the
intended referent and the speaker's communicative goal given the heard utterance.
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
def L_goal[u: UTT, r: OBJ](alpha, lex: ..., m: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(g in UTT, wpp=at(lex, g, r) + {EPS}),
        speaker: chooses(
            u in UTT,
            wpp=(at(lex, u, r) + {EPS}) * exp(alpha * log(at(m, u, g) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    l0 = ctx.lex / (ctx.lex.sum(axis=1, keepdims=True) + EPS)
    m = l0 @ ctx.lex.T
    heard = L_goal(params["alpha"], ctx.lex, m)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
