"""RSA depth-2 pragmatic listener with utterance extension costs, evaluative salience prior, and softmax listener choice rule.

Refining costly_valence_l2_listener: listeners interpret referential and evaluative
descriptions by reasoning at depth 2 while inverting a speaker who balances
communicative informativeness against utterance production costs, and select
referents using a softmax choice rule with decision rationality beta rather than
pure probability matching.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "w_valence": dist.Normal(0.0, 2.0),
    "w_extension": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., weights: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex, prior) + {EPS}) + vec(weights, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, beta, lex: ..., prior: ..., weights: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, prior, weights) + {EPS}) + vec(weights, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        (params["w_features"] + params["w_valence"] * ctx.valence) * ctx.feature_count
        + params["w_familiar"] * ctx.familiarization
    )
    weights = params["w_extension"] * (1.0 - ctx.is_sink) * (jnp.sum(ctx.lex, axis=1) - 1.0)
    heard = L2(params["alpha"], params["beta"], ctx.lex, prior, weights)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
