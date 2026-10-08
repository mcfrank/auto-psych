"""Aspect goal listener: pragmatic inference over a speaker communicating object aspects.

Listeners assume that a speaker communicates to convey an informative aspect or feature of
an object rather than specifying its full identity. When describing a referent, the speaker
selects an aspect from among the object's features and chooses an expression that maximizes
informativeness about that aspect. Pragmatic listeners jointly infer the intended referent
and the speaker's communicative goal, penalizing objects with additional unmentioned features
because multi-featured referents divide the speaker's communicative goals across multiple
candidate aspects. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0_aspect[u: UTT, g: UTT](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[at(lex, g, listener.r) > 0]


@memo
def L_qud[u: UTT, r: OBJ](alpha, beta, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(g in UTT, wpp=at(lex, g, r) + {EPS}),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0_aspect[u, g](lex) + {EPS})) + {EPS},
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(params["w_familiar"] * ctx.familiarization)
    heard = L_qud(params["alpha"], params["beta"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
