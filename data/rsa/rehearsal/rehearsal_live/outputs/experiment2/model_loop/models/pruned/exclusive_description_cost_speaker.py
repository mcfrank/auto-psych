"""Pragmatic listener reasoning about an exclusive-description-cost speaker.

Speakers in reference games actively avoid producing non-exclusive descriptions
that are shared with visual competitors, incurring a cognitive production cost
whenever they use an ambiguous referring expression instead of an exclusive
identifying label. Pragmatic listeners invert this ambiguity-averse speaker,
inferring that when an ambiguous expression is produced, the speaker lacked any
exclusive alternative for the intended referent. On uninformative trials where
no informative word is uttered, listener expectations reflect that speakers
prefer to communicate about referents that possess an exclusive identifying
description.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "c_ambig": dist.Normal(0.0, 2.0),
    "w_prior": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, c_ambig, lex: ..., non_exclusive: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - c_ambig * vec(non_exclusive, u)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_exclusivity(ctx):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    extension = jnp.sum(real_lex, axis=1)

    is_exclusive = (
        (extension > 0.5) & (extension < 1.5)
    ).astype(jnp.float32) * real_words
    non_exclusive = (1.0 - is_exclusive) * real_words
    has_exclusive = jnp.max(is_exclusive[:, None] * real_lex, axis=0)

    return non_exclusive, has_exclusive


def choice_probs(params, ctx):
    non_exclusive, has_exclusive = compute_exclusivity(ctx)
    prior = softmax_prior(params["w_prior"] * has_exclusive)
    heard = L1(params["alpha"], params["c_ambig"], ctx.lex, non_exclusive)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
