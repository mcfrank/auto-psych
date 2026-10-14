"""RSA depth-2 pragmatic listener with bounded capacity, feature redundancy discounting, and contextual end-point salience.

Refining bounded_capacity_redundancy_l2 with contextual end-point salience
from endpoint_salience_l2: listeners and speakers anchor default referential
expectations at the boundaries of the contextual feature complexity distribution
(the minimal baseline object and the maximal exemplar), disfavoring intermediate
objects that fall strictly between these poles. Capacity-limited listeners invert
a redundancy-discounted speaker under finite information-processing capacity kappa.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "kappa": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_empty": dist.Normal(0.0, 1.0),
    "w_endpoint": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_redundancy": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_redundancy(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Compute contextual redundancy for each utterance via pairwise Jaccard overlap."""
    real_lex = lex * (1.0 - is_sink)[:, None]
    intersection = jnp.dot(real_lex, real_lex.T)
    counts = jnp.sum(real_lex, axis=1)
    union = counts[:, None] + counts[None, :] - intersection
    jaccard = jnp.where(union > 0, intersection / (union + 1e-10), 0.0)
    jaccard = jnp.where(jnp.eye(lex.shape[0]) > 0, 1.0, jaccard)
    sink_mask = (
        is_sink[:, None] * (1.0 - is_sink[None, :])
        + (1.0 - is_sink[:, None]) * is_sink[None, :]
    )
    jaccard = jnp.where(sink_mask > 0, 0.0, jaccard)
    return jnp.sum(jaccard, axis=1)


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def S1[u: UTT, r: OBJ](alpha, lex: ..., alt_weight: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r)
        * vec(alt_weight, u)
        * exp(alpha * log(L0[u, r](lex) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L1[u: UTT, r: OBJ](kappa, lex: ..., prior: ..., s1: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * vec(prior, r)
        * exp(kappa * log(at(s1, u, r) + {EPS})),
    )
    return Pr[listener.r == r]


@memo
def S2[u: UTT, r: OBJ](alpha, lex: ..., l1: ..., alt_weight: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r)
        * vec(alt_weight, u)
        * exp(alpha * log(at(l1, u, r) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L2[u: UTT, r: OBJ](kappa, lex: ..., prior: ..., s2: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * vec(prior, r)
        * exp(kappa * log(at(s2, u, r) + {EPS})),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    is_empty = jnp.where(ctx.feature_count == 0, 1.0, 0.0)

    k_min = jnp.min(ctx.feature_count)
    k_max = jnp.max(ctx.feature_count)
    is_inter = jnp.where(
        (ctx.feature_count > k_min) & (ctx.feature_count < k_max), 1.0, 0.0
    )
    is_endpoint = 1.0 - is_inter

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_empty"] * is_empty
        + params["w_endpoint"] * is_endpoint
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    redundancy = compute_redundancy(ctx.lex, ctx.is_sink)
    alt_weight = jnp.exp(
        -params["w_redundancy"] * jnp.log(jnp.maximum(redundancy, 1.0))
    )

    s1 = S1(params["alpha"], ctx.lex, alt_weight)
    l1 = L1(params["kappa"], ctx.lex, prior, s1)
    s2 = S2(params["alpha"], ctx.lex, l1, alt_weight)
    heard = L2(params["kappa"], ctx.lex, prior, s2)[ctx.utterance]

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
