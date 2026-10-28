"""Contrastive prototype categorization and Euclidean psychological distance at depth 2.

Listeners and speakers evaluate candidate words and referents through Tverskyan
contrastive diagnosticity and Euclidean psychological distance:
1. A word's literal prototype fit narrows toward minimal-feature objects in
   proportion to the fraction of display distractors it eliminates, while
   non-contrastive words that apply to every object in the display attenuate the
   speaker's pragmatic sensitivity to residual ambiguity.
2. Perceptual salience over objects combines sublinear Euclidean distance in
   feature space with singleton uniqueness, feature complexity modulated by
   evaluative valence framing, and color saturation contrast.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](g_lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(g_lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., g_lex: ..., prior: ..., diag_w: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * vec(diag_w, u) * log(L0[u, r](g_lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., g_lex: ..., prior: ..., diag_w: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * vec(diag_w, u)
                * log(L1[u, r](alpha, lex, g_lex, prior, diag_w) + {EPS})
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    real_mask = 1.0 - ctx.is_sink
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * real_mask[:, None, None], axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)

    # Sublinear Euclidean distance in binary feature space: sqrt(Hamming distance)
    distinct = jnp.sum(jnp.sqrt(pair_dist), axis=1)

    effective_w_features = (
        params["w_features"] + ctx.valence * params["w_valence"]
    )

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + effective_w_features * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    n_obj = ctx.lex.shape[1]
    n_true = jnp.sum(ctx.lex, axis=-1)
    elim_rate = ((n_obj - n_true) / jnp.maximum(n_obj - 1.0, 1.0)) * real_mask
    has_contrast = jnp.where(n_obj - n_true > 0.5, 1.0, 0.0) * real_mask

    gamma = params["gamma"]
    fc = jnp.maximum(ctx.feature_count, 1.0)
    g_lex = ctx.lex * jnp.exp(-gamma * elim_rate[:, None] * (fc[None, :] - 1.0))
    diag_w = jnp.exp(-0.5 * gamma * (1.0 - has_contrast) * real_mask)

    heard = L2(params["alpha"], ctx.lex, g_lex, prior, diag_w)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
