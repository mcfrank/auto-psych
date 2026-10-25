# Ported from literal_listener.py by src.rsa_open.port, rule (a) (docs/auto_open/PLAN.md §3.1).
"""Literal listener (depth 0).

The listener picks uniformly among the objects the heard word is literally
true of; with no informative word, uniformly among all objects. A lapse
parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


def listener(params, ctx):
    heard = L0(ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return jnp.where(ctx.is_prior > 0, uniform, heard)
