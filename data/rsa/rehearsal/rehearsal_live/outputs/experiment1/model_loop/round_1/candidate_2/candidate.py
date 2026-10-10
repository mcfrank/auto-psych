"""Mutual exclusivity heuristic listener.

Listeners resolve ambiguous referring expressions using the mutual exclusivity
principle: when an uttered feature applies to multiple candidate referents,
listeners penalize any referent that possesses an exclusive, contextually
unique feature that the speaker could have used instead. On uninformative prior
trials without an informative word, choice is uniform. A lapse parameter mixes
in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_me[u: UTT, r: OBJ](beta, lex: ..., exclusive_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * vec(exclusive_count, r)),
    )
    return Pr[listener.r == r]


def compute_exclusive_count(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    ext = jnp.sum(real_lex, axis=1, keepdims=True)
    is_exclusive_feat = (ext == 1.0).astype(jnp.float32) * real_words
    return jnp.sum(real_lex * is_exclusive_feat, axis=0)


def choice_probs(params, ctx):
    exclusive_count = compute_exclusive_count(ctx)
    heard = L_me(params["beta"], ctx.lex, exclusive_count)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard),
        params["lapse"],
    )
