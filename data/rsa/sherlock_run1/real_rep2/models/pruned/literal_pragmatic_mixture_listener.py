"""Literal-pragmatic mixture listener: population heterogeneity over listener types.

The population of listeners is cognitively heterogeneous, consisting of a mixture of
literal and pragmatic reasoning types rather than a uniform reasoning depth. While
pragmatic listeners engage in counterfactual social reasoning by softly maximizing
over the communicative beliefs of an informative speaker, literal listeners simply
choose among candidate referents that satisfy the literal semantics of the utterance.
Aggregate choices across participants reflect the mixture proportion of these discrete
listener types in the population.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 1.0),
    "p_pragmatic": dist.Beta(1.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, gamma, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(gamma * Pr[speaker.r == r]))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    l0 = L0(ctx.lex, prior)[ctx.utterance]
    l1 = L1(params["alpha"], params["gamma"], ctx.lex, prior)[ctx.utterance]
    heard = params["p_pragmatic"] * l1 + (1.0 - params["p_pragmatic"]) * l0
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
