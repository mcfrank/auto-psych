"""Pragmatic listener inverting a speaker governed by mutual validity across cue validity and category validity."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(2.2, 0.4),
    "w_cat": dist.LogNormal(0.0, 0.5),
    "epsilon": dist.Beta(1.0, 5.0),
    "lambda_sim": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_extremity": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, epsilon, lex: ..., prior: ..., util: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            m in UTT,
            wpp=at(lex, m, r) * exp(alpha * at(util, m, r)),
        ),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * ((1.0 - epsilon) * (u == m) + epsilon),
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

    # Cue validity: referential specificity in visual context
    sim = jnp.exp(-params["lambda_sim"] * total_dist)
    conf = jnp.einsum("uc, cr -> ur", ctx.lex, sim)
    cue_validity = ctx.lex / (conf + EPS)

    # Category validity: referential exhaustivity (penalizing excess unmentioned features)
    counts = jnp.where(ctx.lex > 0.0, ctx.feature_count[None, :], 999.0)
    min_counts = jnp.min(counts, axis=1, keepdims=True)
    excess = jnp.maximum(ctx.feature_count[None, :] - min_counts, 0.0)
    cat_validity = jnp.where(ctx.is_sink[:, None] > 0.0, 1.0, ctx.lex / (1.0 + excess))

    # Mutual validity utility
    util = jnp.log(cue_validity + EPS) + params["w_cat"] * jnp.log(cat_validity + EPS)
    max_u = jnp.max(jnp.where(ctx.lex > 0, util, -1e9), axis=0, keepdims=True)
    util_stable = util - max_u

    heard = L1(
        params["alpha"], params["epsilon"], ctx.lex, prior, util_stable
    )[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
