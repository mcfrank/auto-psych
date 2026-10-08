"""RSA depth-2 listener with feature simplicity prior and speaker ambiguity cost.

Refinement of feature_prior_l2: Communicators maintain an inductive prior favoring
simpler referents with fewer distinguishing features, while speakers also incur an
explicit communicative penalty when uttering ambiguous descriptors that apply to
multiple referents in the scene. Listeners reason recursively about this ambiguity-averse,
simplicity-sensitive speaker at depth 2.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost": dist.LogNormal(0.0, 1.0),
    "feature_weight": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, cost, lex: ..., costs: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}) - cost * vec(costs, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, cost, lex: ..., costs: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, cost, lex, costs, prior) + {EPS}) - cost * vec(costs, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(params["feature_weight"] * ctx.feature_count)
    ambiguity = (1.0 - ctx.is_sink) * jnp.maximum(0.0, jnp.sum(ctx.lex, axis=1) - 1.0)
    heard = L2(params["alpha"], params["cost"], ctx.lex, ambiguity, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
