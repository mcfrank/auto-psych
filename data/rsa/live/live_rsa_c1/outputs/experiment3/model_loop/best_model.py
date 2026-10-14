"""Pragmatic listener inverting a communicative speaker under feature surprisal and ensemble grouping expectations."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(2.0, 0.3),
    "p_aspect": dist.Beta(2.0, 2.0),
    "epsilon": dist.Beta(1.0, 5.0),
    "lambda_sim": dist.LogNormal(0.0, 0.5),
    "w_surprisal": dist.Normal(0.0, 1.0),
    "w_group": dist.Normal(1.0, 0.5),
    "w_empty": dist.Normal(0.0, 1.0),
    "w_focal": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}

GOALS = (0, 1)


@memo
def L1[u: UTT, r: OBJ](
    alpha, p_aspect, epsilon, lex: ..., prior: ..., util_ident: ..., util_aspect: ...
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            g in GOALS,
            wpp=(1.0 - p_aspect) * (1.0 - g) + p_aspect * g,
        ),
        speaker: chooses(
            m in UTT,
            wpp=at(lex, m, r)
            * exp(
                alpha
                * (
                    (1.0 - g) * at(util_ident, m, r)
                    + g * at(util_aspect, m, r)
                )
            ),
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
    group_discount = -jnp.log(jnp.maximum(duplicate_count, 1.0))

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    feat_freq = jnp.mean(real_lex, axis=1, keepdims=True)
    feat_surprisal = -jnp.log(feat_freq + 1e-4)
    obj_surprisal = jnp.sum(real_lex * feat_surprisal, axis=0)

    max_count = jnp.max(ctx.feature_count)
    is_focal = jnp.where(
        (ctx.feature_count == max_count) & (ctx.feature_count > 0.0),
        1.0,
        0.0,
    )
    is_empty = jnp.where(ctx.feature_count == 0.0, 1.0, 0.0)

    prior = softmax_prior(
        params["w_surprisal"] * obj_surprisal
        + params["w_group"] * group_discount
        + params["w_empty"] * is_empty
        + params["w_focal"] * is_focal
    )

    sim = jnp.exp(-params["lambda_sim"] * total_dist)
    conf = jnp.einsum("uc, cr -> ur", ctx.lex, sim)
    l0_sim = ctx.lex / (conf + EPS)
    util_ident = jnp.log(l0_sim + EPS)

    counts = jnp.where(ctx.lex > 0.0, ctx.feature_count[None, :], 999.0)
    min_counts = jnp.min(counts, axis=1, keepdims=True)
    excess = jnp.maximum(ctx.feature_count[None, :] - min_counts, 0.0)
    util_aspect = -jnp.log(1.0 + excess)

    heard = L1(
        params["alpha"],
        params["p_aspect"],
        params["epsilon"],
        ctx.lex,
        prior,
        util_ident,
        util_aspect,
    )[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
