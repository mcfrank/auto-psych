"""Pragmatic listener at depth 2 with a contextual feature surprisal topic prior.

Listeners expect communication to focus on entities with high contextual information
content, believing speakers preferentially talk about objects possessing rare or
surprising visual features that violate scene-level expectations. An object's
surprisal is the sum of negative log contextual frequencies across all non-sink
features it possesses. Listeners invert an informative speaker whose choice of
referent is biased by this feature surprisal prior through recursive pragmatic
reasoning, while on uninformative prior trials choices directly reflect this
information-theoretic salience. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_surprisal": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def object_feature_surprisal(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Cumulative contextual feature surprisal (information content) for each object."""
    features = lex * (1.0 - is_sink)[:, None]
    n_obj = lex.shape[1]
    counts = jnp.sum(features, axis=1)
    p = counts / n_obj
    surp = -jnp.log(jnp.maximum(p, 1e-6))
    real_surp = jnp.where(is_sink > 0, 0.0, surp)
    return jnp.sum(features * real_surp[:, None], axis=0)


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    surp = object_feature_surprisal(ctx.lex, ctx.is_sink)
    prior = softmax_prior(params["w_surprisal"] * surp)
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
