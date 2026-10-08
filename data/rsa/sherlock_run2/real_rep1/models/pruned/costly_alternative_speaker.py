"""RSA pragmatic listener reasoning about a speaker penalized for suboptimal alternatives.

In standard RSA, speakers consider all words true of an object with no cost. Under
this model, speakers incur a communicative penalty for producing an utterance that is
less informative than the best alternative available for that referent. Listeners invert
this alternative-sensitive speaker, knowing that hearing a shared word implies the speaker
had no more informative word available.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost_suboptimal": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, cost_suboptimal, lex: ..., gap: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - cost_suboptimal * at(gap, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    col_sums = ctx.lex.sum(axis=1, keepdims=True)
    l0 = ctx.lex / jnp.where(col_sums > 0, col_sums, 1.0)
    max_info = jnp.max(l0, axis=0, keepdims=True)
    gap = jnp.maximum(0.0, max_info - l0) * (ctx.lex > 0)
    heard = L1(
        params["alpha"], params["cost_suboptimal"], ctx.lex, gap
    )[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
