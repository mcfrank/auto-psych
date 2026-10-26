"""Feature uncertainty listener.

Listeners interpret referring expressions by recognizing that word-to-feature mappings are
subject to uncertainty rather than deterministic one-to-one bindings, inverting a rational
speaker while marginalizing across the visual features present in the scene. A referent
possessing multiple distinctive features remains robustly identifiable under this feature
uncertainty even when a shared descriptor is used, resolving the over-penalization of
overspecified competitors. On uninformative displays without an informative word, referent
expectations are governed by visual singleton salience and contextual distinctiveness.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "logit_theta": dist.Normal(-1.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
}


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


def listener(params, ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words

    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    dist = jnp.sum(diff, axis=0)

    # Perceptual prior: discrete singleton salience and continuous contextual distinctiveness
    duplicate_count = jnp.sum(dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)
    distinct = jnp.sum(dist, axis=1)
    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
    )

    # Pragmatic listener table across all utterances: shape (N_UTT, N_OBJ)
    table = L1(params["alpha"], ctx.lex, prior)

    # Feature uncertainty: marginalize over intended visual features
    rw = 1.0 - ctx.is_sink
    feat_weights = rw / jnp.maximum(jnp.sum(rw), 1.0)
    marginal = jnp.sum(table * feat_weights[:, None], axis=0)

    theta = jax.nn.sigmoid(params["logit_theta"])
    focal = table[ctx.utterance]
    heard = (1.0 - theta) * focal + theta * marginal

    return jnp.where(ctx.is_prior > 0, prior, heard)
