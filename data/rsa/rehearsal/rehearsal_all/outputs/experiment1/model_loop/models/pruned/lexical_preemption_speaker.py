"""Pragmatic listener reasoning about a speaker with lexical preemption.

Speakers evaluate candidate referring expressions relative to the best available
description for their intended referent: candidate words that are ambiguous or
partially shared incur a preemption penalty proportional to how much less informative
they are compared to the referent's best available word. When an intended referent
cannot be uniquely named, no superior alternative exists, so the shared word is produced
without preemption penalty. Pragmatic listeners invert this preemption-sensitive speaker,
attributing ambiguous utterances to referents that lacked distinctive naming options.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_preempt": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_preempt, lex: ..., preemption: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - w_preempt * at(preemption, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_preemption_and_prior(ctx, w_preempt):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    word_counts = jnp.sum(real_lex, axis=1, keepdims=True)
    L0_mat = real_lex / jnp.maximum(word_counts, 1.0)
    best_L0 = jnp.max(L0_mat, axis=0, keepdims=True)
    preemption = (best_L0 - L0_mat) * real_lex
    nameability = best_L0[0]
    prior = softmax_prior(w_preempt * nameability)
    return preemption, prior


def choice_probs(params, ctx):
    preemption, prior = compute_preemption_and_prior(ctx, params["w_preempt"])
    heard = L1(params["alpha"], params["w_preempt"], ctx.lex, preemption)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
