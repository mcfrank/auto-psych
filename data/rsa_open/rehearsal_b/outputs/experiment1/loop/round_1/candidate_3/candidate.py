"""Pragmatic listener reasoning about a cost-sensitive speaker with unnameable referent fallback.

Speakers select referring expressions to maximize communicative informativeness while
penalizing utterance ambiguity, falling back to silence when a referent cannot be named.
Pragmatic listeners invert this cost-sensitive speaker, attributing informative words
to referents that lack more specific alternatives and attributing silence to unnameable
or minimally featured referents.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_cost": dist.Normal(0.0, 1.0),
    "lambda_unnameable": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_cost, lex: ..., prior: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - w_cost * vec(cost, u)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, w_cost, lex: ..., prior: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, w_cost, lex, prior, cost) + {EPS})
                - w_cost * vec(cost, u)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    ext = jnp.sum(ctx.lex * (1.0 - ctx.is_sink[:, None]), axis=-1) / ctx.lex.shape[-1]
    prior = softmax_prior(-params["lambda_unnameable"] * ctx.feature_count)
    heard = L2(params["alpha"], params["w_cost"], ctx.lex, prior, ext)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
