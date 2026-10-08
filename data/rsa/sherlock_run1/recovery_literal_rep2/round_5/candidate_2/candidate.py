"""Pragmatic listener reasoning about a speaker with costly distinctive features.

Speakers in reference games balance communicative informativeness against the
cognitive production and retrieval cost of referring expressions: using a uniquely
distinguishing feature requires finer perceptual discrimination and specialized
lexical access, incurring a production cost relative to shared, accessible descriptors.
The pragmatic listener inverts this cost-sensitive speaker under a softmax decision rule:
hearing a shared word does not strongly rule out candidate referents that possess
unuttered unique features, because the speaker had a cost disincentive against
producing those distinctive alternatives.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "cost": dist.Normal(0.0, 1.0),
    "beta": dist.LogNormal(-2.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](cost, beta, lex: ..., is_unique: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(log(L0[u, r](lex) + {EPS}) - cost * vec(is_unique, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(
        r in OBJ,
        wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    col_sums = jnp.sum(ctx.lex, axis=1)
    is_unique = jnp.where((col_sums == 1.0) & (ctx.is_sink == 0.0), 1.0, 0.0)
    heard = L1(params["cost"], params["beta"], ctx.lex, is_unique)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
