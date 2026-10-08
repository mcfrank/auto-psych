"""Prototype typicality listener.

When anticipating what a speaker will refer to before hearing an informative
description (or when choosing on prior trials), listeners expect the speaker to
talk about the most typical or prototypical exemplar in the visual display.
Grounded in prototype theory and family resemblance, an object's contextual
typicality is determined by the number of visual features it shares with the
other objects in the context. Pragmatic listeners invert a communicative speaker
who operates over this contextual typicality prior, favoring prototypical referents
with shared features over atypical outliers.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_typical": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    shared_counts = jnp.sum(real_lex, axis=1, keepdims=True) - real_lex
    typicality = jnp.sum(real_lex * shared_counts, axis=0)
    prior = softmax_prior(params["w_typical"] * typicality)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
