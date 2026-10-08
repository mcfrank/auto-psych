"""RSA pragmatic listener with a distinctiveness-based salience prior.

Listeners interpret referential descriptions by maintaining a prior expectation
that speakers refer to distinctive objects—those whose features contrast with
the visual context by being unique or narrowly shared, rather than broadly shared.
Before hearing an informative utterance, listeners assign higher prior probability
to objects possessing distinctive, specific features; this distinctiveness prior
is shared as common ground across the simulated speaker and listener.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    feature_freq = jnp.sum(real_lex, axis=1)
    specificity = jnp.where(feature_freq > 0, 1.0 / jnp.maximum(feature_freq, 1.0), 0.0) * (1.0 - ctx.is_sink)
    distinctiveness = jnp.sum(real_lex * specificity[:, None], axis=0)

    prior = softmax_prior(
        params["w_distinct"] * distinctiveness + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
