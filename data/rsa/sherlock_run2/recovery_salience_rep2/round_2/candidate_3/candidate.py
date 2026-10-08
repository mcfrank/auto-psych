"""Feature uncertainty listener.

Listeners experience lexical uncertainty regarding which visual feature a word
picks out, assigning a small probability that the speaker is referring to an
alternative feature present in the visual context.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "uncertainty": dist.Beta(1.0, 9.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](soft_lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(soft_lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, soft_lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(soft_lex, u, r) * exp(alpha * log(L0[u, r](soft_lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, soft_lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(soft_lex, u, r) * exp(alpha * log(L1[u, r](alpha, soft_lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    n_real = jnp.sum(1.0 - ctx.is_sink)
    denom = jnp.maximum(1.0, n_real - 1.0)
    alt_features = (ctx.feature_count - ctx.lex) / denom
    alt_lex = jnp.where(n_real > 1.0, alt_features, 0.0)
    soft_features = (
        (1.0 - params["uncertainty"]) * ctx.lex + params["uncertainty"] * alt_lex
    )
    soft_lex = jnp.where(ctx.is_sink[:, None] > 0.0, ctx.lex, soft_features)

    heard = L2(params["alpha"], soft_lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
