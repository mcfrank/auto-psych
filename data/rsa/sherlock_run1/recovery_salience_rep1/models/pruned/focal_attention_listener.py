"""Pragmatic listener with focal display attention.

Resource-limited listeners focus visual attention primarily on candidate
referents that match the heard utterance, attenuating the attentional weight of
non-candidate objects in the visual display. When simulating what a speaker
would say, the listener's mental model evaluates the communicative informativeness
of alternative utterances against this attention-weighted display representation,
discounting irrelevant distractors.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "distractor_weight": dist.Beta(1.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., atten: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * (vec(atten, r) + {EPS}))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., atten: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, atten) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    is_cand = jnp.where(ctx.is_prior > 0, 1.0, ctx.lex[ctx.utterance])
    atten = jnp.where(is_cand > 0, 1.0, params["distractor_weight"])
    heard = L1(params["alpha"], ctx.lex, atten)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
