"""Perceptual distinctiveness heuristic listener model.

Listeners do not simulate a communicative speaker using recursive theory of mind.
Instead, they use a visual distinctiveness heuristic: after filtering candidate
objects to those that literally match the uttered word, the listener selects among
them in proportion to each object's overall visual distinctiveness (total feature
contrast / Hamming distance) relative to all other objects in the display. On
uninformative prior trials, listeners choose directly based on this visual
distinctiveness metric.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "w_distinct": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_distinct[u: UTT, r: OBJ](w_distinct, lex: ..., distinct: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(w_distinct * vec(distinct, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))
    heard = L_distinct(params["w_distinct"], ctx.lex, distinct)[ctx.utterance]
    prior = softmax_prior(params["w_distinct"] * distinct)
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
