import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_empty": dist.Normal(0.0, 1.0),
    "w_endpoint": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_perceptual_prior(params, ctx):
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

    return softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_empty"] * is_empty
        + params["w_endpoint"] * is_endpoint
        + params["w_color"] * (1.0 - ctx.grayscale)
    )


def compute_alternative_specificity(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    real_lex = lex * (1.0 - is_sink)[:, None]
    freq = jnp.sum(real_lex, axis=1)
    spec = jnp.where(is_sink > 0, 0.0, 1.0 / jnp.maximum(freq, 1.0))
    total_spec = jnp.sum(real_lex * spec[:, None], axis=0)
    alt_spec = total_spec[None, :] - real_lex * spec[:, None]
    return jnp.maximum(alt_spec, 0.0)


@memo
def L_heur[u: UTT, r: OBJ](beta, lex: ..., alt_spec: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * at(alt_spec, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = compute_perceptual_prior(params, ctx)
    alt_spec = compute_alternative_specificity(ctx.lex, ctx.is_sink)
    heard = L_heur(params["beta"], ctx.lex, alt_spec)[ctx.utterance]
    p_choice = (1.0 - params["lapse"]) * heard + params["lapse"] * prior
    return jnp.where(ctx.is_prior > 0, prior, p_choice)
