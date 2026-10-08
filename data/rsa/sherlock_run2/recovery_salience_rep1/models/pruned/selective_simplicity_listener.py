"""Pragmatic listener combining selective visual attention with referent simplicity.

Refinement of selective_attention_listener: Communicators combine selective visual
attention toward utterance-compatible referents with an inductive preference for
simpler referents with fewer distinguishing features (taken from
feature_simplicity_listener). In L1, the speaker samples referents according to
a softmax prior over object feature counts rather than uniformly, while the
literal listener L0 discounts unattended distractors. On uninformative prior
trials, the listener relies directly on the simplicity prior rather than
guessing uniformly.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "distractor_attention": dist.Beta(1.0, 1.0),
    "simplicity": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., att: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(att, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., att: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, att) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    matching = ctx.lex[ctx.utterance]
    att = jnp.where(matching > 0, 1.0, params["distractor_attention"])
    prior = softmax_prior(params["simplicity"] * ctx.feature_count)
    heard = L1(params["alpha"], ctx.lex, att, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
