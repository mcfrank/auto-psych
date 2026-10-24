"""Causal diagnostic listener.

Listeners interpret referring expressions through causal diagnostic inference
rather than recursive mental simulation, attributing a heard word to the referent
that maximizes causal power relative to the counterfactual production baseline
across display alternatives. Speakers communicate by selecting words that reliably
distinguish the intended target from contextual competitors. On uninformative displays
without an informative word, visual attention defaults to contextually unique
singletons and feature contrast.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
}


@memo
def L_causal[u: UTT, r: OBJ](beta, lex: ..., power: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(beta * at(power, u, r)),
    )
    return Pr[listener.r == r]


def compute_singleton(ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    return jnp.where(duplicate_count == 1, 1.0, 0.0)


def compute_distinct(ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    return jnp.sum(diff, axis=(0, 2))


def compute_causal_power(alpha, ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    ext = jnp.maximum(jnp.sum(real_lex, axis=1, keepdims=True), 1.0)
    l0 = real_lex / ext

    s_unnorm = real_lex * jnp.exp(alpha * jnp.log(l0 + EPS))
    s_denom = jnp.maximum(jnp.sum(s_unnorm, axis=0, keepdims=True), 1e-6)
    p_u_r = s_unnorm / s_denom  # (N_UTT, N_OBJ)

    n_obj = ctx.lex.shape[1]
    total_p_u = jnp.sum(p_u_r, axis=1, keepdims=True)
    p_u_not_r = (total_p_u - p_u_r) / jnp.maximum(n_obj - 1.0, 1.0)

    delta_p = p_u_r - p_u_not_r
    power = jnp.maximum(0.0, delta_p) / jnp.maximum(1e-6, 1.0 - p_u_not_r + 1e-6)
    return power * real_words


def listener(params, ctx):
    is_singleton = compute_singleton(ctx)
    distinct = compute_distinct(ctx)
    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
    )
    power = compute_causal_power(params["alpha"], ctx)
    heard = L_causal(params["beta"], ctx.lex, power)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
