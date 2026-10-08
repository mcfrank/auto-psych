"""Distractor-suppression pragmatic listener.

Listeners operate with resource-limited attention to the display: once an
utterance is heard, visual attention is focused on candidate objects that match
the utterance, while non-matching distractor objects receive attenuated
attention when the listener evaluates the communicative informativeness of
alternative referring expressions.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "distractor_attention": dist.Beta(2.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., obj_attention: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * (vec(obj_attention, r) + {EPS}))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., obj_attention: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, obj_attention) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    matches_heard = ctx.lex[ctx.utterance]
    attention = jnp.where(matches_heard > 0, 1.0, params["distractor_attention"])
    attention = jnp.where(ctx.is_prior > 0, 1.0, attention)
    heard = L1(params["alpha"], ctx.lex, attention)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
