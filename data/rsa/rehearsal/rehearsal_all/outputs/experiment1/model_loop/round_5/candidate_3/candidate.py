"""Pragmatic listener reasoning about a speaker with communicative affordance.

Speakers in reference games preferentially choose target referents that possess
high communicative affordance, favoring objects that can be clearly and unambiguously
identified using true referring expressions. Pragmatic listeners model this target-selection
preference, expecting speakers to talk about referents with higher communicative clarity
rather than ambiguous or hard-to-describe items. Pragmatic listeners invert this
affordance-sensitive speaker at depth two to resolve referring expressions and predict
spontaneous choices on uninformative trials.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_afford": dist.Normal(0.0, 2.0),
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
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_affordance_prior(ctx, alpha, w_afford):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    word_counts = jnp.sum(real_lex, axis=1, keepdims=True)
    l0_matrix = real_lex / jnp.maximum(word_counts, 1.0)

    word_weights = real_lex * jnp.exp(alpha * jnp.log(l0_matrix + EPS))
    affordance = jnp.sum(word_weights, axis=0)
    log_affordance = jnp.log(affordance + 1.0)
    return softmax_prior(w_afford * log_affordance)


def choice_probs(params, ctx):
    prior = compute_affordance_prior(ctx, params["alpha"], params["w_afford"])
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
