"""Pragmatic listener inverting a speaker with Gricean specificity deficit."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta_deficit": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_extremity": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, beta_deficit, lex: ..., prior: ..., base_util: ..., deficit: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * at(base_util, u, r) - beta_deficit * at(deficit, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
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

    min_count = jnp.min(ctx.feature_count)
    max_count = jnp.max(ctx.feature_count)
    midpoint = 0.5 * (min_count + max_count)
    extremity = jnp.abs(ctx.feature_count - midpoint)

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_extremity"] * extremity
        + params["w_color"] * (1.0 - ctx.grayscale)
        + params["w_familiar"] * ctx.familiarization
    )

    non_sink = 1.0 - ctx.is_sink
    ext = jnp.sum(ctx.lex * non_sink[:, None], axis=1)
    spec = jnp.where(non_sink > 0, 1.0 / (ext + EPS), 0.0)

    true_spec = spec[:, None] * ctx.lex * non_sink[:, None]
    max_spec = jnp.max(true_spec, axis=0)
    deficit = jnp.maximum(0.0, max_spec[None, :] - spec[:, None]) * non_sink[:, None]

    sink_count = jnp.sum(ctx.is_sink[:, None] * ctx.lex)
    l0 = jnp.where(
        ctx.is_sink[:, None] > 0,
        ctx.lex / (sink_count + EPS),
        spec[:, None] * ctx.lex,
    )
    base_util = jnp.log(l0 + EPS)

    heard = L1(
        params["alpha"], params["beta_deficit"], ctx.lex, prior, base_util, deficit
    )[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
