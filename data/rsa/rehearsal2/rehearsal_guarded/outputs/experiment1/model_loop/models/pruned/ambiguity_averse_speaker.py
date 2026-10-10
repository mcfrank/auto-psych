"""Pragmatic listener reasoning about an ambiguity-averse speaker.

Speakers in reference games actively avoid referring expressions that are ambiguous
or shared with other objects in the context. Candidate words are penalized
proportionally to their degree of non-exclusivity across the display. Pragmatic
listeners invert this ambiguity-averse speaker, expecting uttered words to target
referents for which the expression was maximally specific, and expecting
ambiguous referents to be dispreferred a priori.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_ambig": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_ambig, lex: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - w_ambig * vec(cost, u)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_cost_and_prior(ctx, w_ambig):
    real_words = 1.0 - ctx.is_sink
    n_obj = ctx.lex.shape[1]
    n_u = jnp.sum(ctx.lex * real_words[:, None], axis=1)
    denom = jnp.maximum(n_obj - 1.0, 1.0)
    ambig = real_words * jnp.clip((n_u - 1.0) / denom, 0.0, 1.0)

    real_lex = ctx.lex * real_words[:, None]
    word_costs = jnp.where(real_lex > 0, ambig[:, None], 1e5)
    has_any_word = jnp.sum(real_lex, axis=0) > 0
    min_cost = jnp.where(has_any_word, jnp.min(word_costs, axis=0), 1.0)
    prior = softmax_prior(-w_ambig * min_cost)
    return ambig, prior


def choice_probs(params, ctx):
    cost, prior = compute_cost_and_prior(ctx, params["w_ambig"])
    heard = L1(params["alpha"], params["w_ambig"], ctx.lex, cost)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
