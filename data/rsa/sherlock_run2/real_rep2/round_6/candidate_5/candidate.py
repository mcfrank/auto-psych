"""Pragmatic listener at depth 2 with perceptual isolation, feature surprisal prior, graded semantics, and utterance extension costs."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_surprisal": dist.Normal(0.0, 1.0),
    "gamma": dist.Normal(0.0, 1.0),
    "cost_weight": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def object_isolation(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Minimum Hamming distance of each object to any other object in the display."""
    features = lex * (1.0 - is_sink)[:, None]
    diff = jnp.abs(features[:, :, None] - features[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    n_obj = lex.shape[1]
    eye = jnp.eye(n_obj) * 1e5
    return jnp.min(pair_dist + eye, axis=1)


def object_feature_surprisal(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Cumulative contextual feature surprisal (information content) for each object."""
    features = lex * (1.0 - is_sink)[:, None]
    n_obj = lex.shape[1]
    counts = jnp.sum(features, axis=1)
    p = counts / n_obj
    surp = -jnp.log(jnp.maximum(p, 1e-6))
    real_surp = jnp.where(is_sink > 0, 0.0, surp)
    return jnp.sum(features * real_surp[:, None], axis=0)


def compute_graded_lex(lex: jnp.ndarray, feature_count: jnp.ndarray, is_sink: jnp.ndarray, gamma) -> jnp.ndarray:
    """Compute graded truth values where applicability diminishes with additional features."""
    fc = jnp.maximum(feature_count, 1.0)
    precision = jnp.exp(-gamma * (fc - 1.0))
    real_truth = lex * precision[None, :]
    return jnp.where(is_sink[:, None] > 0, lex, real_truth)


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * (log(L0[u, r](lex) + {EPS}) - vec(cost, u))),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * (log(L1[u, r](alpha, lex, prior, cost) + {EPS}) - vec(cost, u))),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    graded_lex = compute_graded_lex(ctx.lex, ctx.feature_count, ctx.is_sink, params["gamma"])
    dist_vec = object_isolation(ctx.lex, ctx.is_sink)
    surp_vec = object_feature_surprisal(ctx.lex, ctx.is_sink)
    prior = softmax_prior(params["w_distinct"] * dist_vec + params["w_surprisal"] * surp_vec)
    ext = jnp.sum(ctx.lex, axis=-1) / ctx.lex.shape[-1]
    cost = params["cost_weight"] * ext
    heard = L2(params["alpha"], graded_lex, prior, cost)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
