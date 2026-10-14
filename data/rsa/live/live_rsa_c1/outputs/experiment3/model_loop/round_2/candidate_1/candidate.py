"""Pragmatic listener with referent expectations grounded in contextual atypicality."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(2.0, 0.3),
    "epsilon": dist.Beta(1.0, 5.0),
    "lambda_sim": dist.LogNormal(0.0, 1.0),
    "w_pos_atypical": dist.Normal(0.0, 1.0),
    "w_neg_atypical": dist.Normal(0.0, 1.0),
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

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    proto = jnp.mean(real_lex, axis=1, keepdims=True)
    pos_dev = jnp.sum(real_lex * (1.0 - proto), axis=0)
    non_sink_mask = 1.0 - ctx.is_sink[:, None]
    neg_dev = jnp.sum((non_sink_mask - real_lex) * proto, axis=0)

    prior = softmax_prior(
        params["w_pos_atypical"] * pos_dev
        + params["w_neg_atypical"] * neg_dev
        + params["w_color"] * (1.0 - ctx.grayscale)
        + params["w_familiar"] * ctx.familiarization
    )

    sim = jnp.exp(-params["lambda_sim"] * total_dist)
    conf = jnp.einsum("uc, cr -> ur", ctx.lex, sim)
    l0_sim = ctx.lex / (conf + EPS)
    util = jnp.log(l0_sim + EPS)

    heard = L1(
        params["alpha"], params["epsilon"], ctx.lex, prior, util
    )[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
