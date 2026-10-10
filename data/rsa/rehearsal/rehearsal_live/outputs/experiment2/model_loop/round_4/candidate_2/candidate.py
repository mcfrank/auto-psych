"""Literal-pragmatic population mixture listener.

Listeners in a reference game differ in their depth of recursive social reasoning:
the participant population is a mixture of literal listeners, who select uniformly
among referents satisfying the literal truth conditions of the uttered word, and
pragmatic listeners, who reason counterfactually about speaker alternatives and
baseline communicative expectations. When no informative expression is uttered,
literal listeners guess uniformly, whereas pragmatic listeners expect speakers
to communicate about descriptive, feature-bearing referents. Observed choices across
the population reflect the proportion of pragmatic versus literal responders.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "p_pragmatic": dist.Beta(2.0, 2.0),
    "w_prior": dist.Normal(0.0, 1.0),
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


def choice_probs(params, ctx):
    prior_prag = softmax_prior(params["w_prior"] * ctx.feature_count)
    uniform = jnp.full_like(prior_prag, 1.0 / prior_prag.shape[0])

    l0 = L0(ctx.lex)[ctx.utterance]
    l1 = L1(params["alpha"], ctx.lex, prior_prag)[ctx.utterance]

    p_prag = params["p_pragmatic"]
    heard = p_prag * l1 + (1.0 - p_prag) * l0
    prior = p_prag * prior_prag + (1.0 - p_prag) * uniform

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard),
        params["lapse"],
    )
