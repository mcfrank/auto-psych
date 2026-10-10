"""Mutual exclusivity listener.

Listeners resolve referential ambiguity using a mutual exclusivity constraint based
on alternative label availability. When hearing a word that applies to multiple
referents, listeners penalize candidate objects in proportion to how uniquely they
are identified by unmentioned alternative features. Consequently, listeners map
the heard expression to the referent that lacks an alternative exclusive name,
defaulting to uniform choice on uninformative prior trials.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_me[u: UTT, r: OBJ](beta, lex: ..., alt_excl: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * at(alt_excl, u, r)),
    )
    return Pr[listener.r == r]


def compute_alt_excl(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    utt_count = jnp.sum(real_lex, axis=1, keepdims=True)
    spec = jnp.where(utt_count > 0, 1.0 / jnp.maximum(utt_count, 1.0), 0.0)
    feat_excl = spec * real_lex
    total_excl = jnp.sum(feat_excl, axis=0, keepdims=True)
    return total_excl - feat_excl


def choice_probs(params, ctx):
    alt_excl = compute_alt_excl(ctx)
    heard = L_me(params["beta"], ctx.lex, alt_excl)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
