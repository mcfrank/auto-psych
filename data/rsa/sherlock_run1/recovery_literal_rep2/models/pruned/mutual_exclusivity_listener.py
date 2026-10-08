"""Pragmatic listener relying on the principle of mutual exclusivity.

When interpreting referring expressions, listeners rely on a principle of mutual
exclusivity rather than recursive mental simulation of the speaker. Upon hearing
a feature word, listeners consider all matching objects and disprefer candidate
referents that possess alternative dedicated features uniquely distinguishing
them in the context. Because an object with an exclusive identifying feature
would be described by that exclusive name, listeners infer that a shared or
ambiguous description is intended for competitor referents lacking dedicated
labels.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "lambda_me": dist.LogNormal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_me[u: UTT, r: OBJ](lambda_me, lex: ..., prior: ..., dedicated: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r) * exp(-lambda_me * vec(dedicated, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(params["w_familiar"] * ctx.familiarization)

    # Feature f is dedicated if exactly one object has it and it's not the sink utterance
    col_sums = jnp.sum(ctx.lex, axis=1)
    is_unique_feat = jnp.where((col_sums == 1.0) & (ctx.is_sink == 0.0), 1.0, 0.0)

    # For the heard utterance u, count dedicated alternative features for object r
    alt_unique = is_unique_feat.at[ctx.utterance].set(0.0)
    dedicated = jnp.sum(ctx.lex * alt_unique[:, None], axis=0)

    heard = L_me(params["lambda_me"], ctx.lex, prior, dedicated)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
