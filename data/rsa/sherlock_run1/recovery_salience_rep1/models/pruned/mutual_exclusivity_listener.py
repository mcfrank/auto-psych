"""Mutual exclusivity listener.

Listeners interpret referential descriptions via a mutual exclusivity heuristic:
when hearing a feature word, they penalize candidate referents that possess
alternative features that uniquely distinguish them in the visual scene. On
displays where candidate objects lack unique distinguishing alternatives,
candidates are evaluated equally. With no informative word, the listener chooses
uniformly. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_me[u: UTT, r: OBJ](beta, lex: ..., unique_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * vec(unique_count, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    utt_counts = lex.sum(axis=1)
    is_unique = jnp.where(utt_counts == 1.0, 1.0, 0.0) * (1.0 - ctx.is_sink)
    unique_count = (lex * is_unique[:, None]).sum(axis=0)

    heard = L_me(params["beta"], ctx.lex, unique_count)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
