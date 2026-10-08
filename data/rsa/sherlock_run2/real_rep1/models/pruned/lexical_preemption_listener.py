"""Lexical preemption listener model.

Listeners interpret referring expressions through the principle of mutual exclusivity
and lexical preemption: when an ambiguous description is heard, candidate referents
that possess an alternative, more distinctive descriptor in the display are perceived
as lexically preempted and thus disfavored in proportion to the specificity of their
unmentioned features. On uninformative trials with no descriptive word, choice is uniform.
A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "lambda_preempt": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_preempt[u: UTT, r: OBJ](lambda_preempt, lex: ..., preempt: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-lambda_preempt * at(preempt, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    counts = jnp.sum(real_lex, axis=1, keepdims=True)
    spec = real_lex / jnp.where(counts > 0, counts, 1.0)
    total_spec = jnp.sum(spec, axis=0, keepdims=True)
    preempt = jnp.maximum(0.0, total_spec - spec)

    heard = L_preempt(params["lambda_preempt"], ctx.lex, preempt)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
