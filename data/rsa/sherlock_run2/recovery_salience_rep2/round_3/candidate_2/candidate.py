"""Alternative feature listener.

Listeners resolve referential ambiguity through a direct alternative-feature
heuristic rather than recursive Theory-of-Mind speaker simulation, choosing
among matching candidate referents with a penalty for objects that possess
alternative distinguishing features.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "gamma": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_alt[u: UTT, r: OBJ](gamma, lex: ..., alt_avail: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-gamma * at(alt_avail, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    feat_counts = jnp.sum(real_lex, axis=1, keepdims=True)
    spec = jnp.where(feat_counts > 0, 1.0 / feat_counts, 0.0)
    total_avail = jnp.sum(real_lex * spec, axis=0, keepdims=True)
    alt_avail = jnp.maximum(0.0, total_avail - (real_lex * spec))

    heard = L_alt(params["gamma"], ctx.lex, alt_avail)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
