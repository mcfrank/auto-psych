"""Pragmatic listener inverting a competitor-reliance communicative speaker.

Speakers evaluate communicative interference from competing objects based on
competitor reliance: a competitor that possesses few alternative descriptive pathways
relies heavily on the shared descriptor and exerts severe communicative interference,
whereas a competitor with multiple alternative descriptors poses minimal threat of
misidentification. A rational pragmatic listener inverts this reliance-sensitive speaker,
recognizing that hearing an ambiguous descriptor strongly points toward the candidate
referent that lacks alternative descriptive options. This mechanism disagrees most sharply
with contrastive similarity models on displays with large feature gaps between a minimal
target and a complex competitor, correctly preventing listeners from over-selecting the
complex competitor.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., util: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * at(util, u, r)),
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

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
        + params["w_familiar"] * ctx.familiarization
    )

    n_obj = ctx.feature_count.shape[0]
    fc_safe = jnp.maximum(ctx.feature_count, 1.0)
    comp_weight = jnp.power(fc_safe, -params["gamma"])
    eye = jnp.eye(n_obj)
    W = eye + (1.0 - eye) * comp_weight[:, None]

    conf = jnp.einsum("uc, cr -> ur", ctx.lex, W)
    l0 = ctx.lex / (conf + EPS)
    util = jnp.log(l0 + EPS)

    heard = L1(params["alpha"], ctx.lex, prior, util)[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
