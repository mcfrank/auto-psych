"""Pragmatic listener at depth 2 reasoning about a speaker with feature-proportional lexical competition."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_contrast": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask
    n_obj = ctx.lex.shape[1]

    # Pairwise feature distance between objects (including grayscale and familiarization contrast)
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    total_dist = pair_dist + gray_diff + fam_diff

    # Smooth nearest-neighbor perceptual contrast (0 for identical duplicates, saturating for distinct items)
    contrast = jnp.sum(1.0 - jnp.exp(-total_dist), axis=1) / jnp.maximum(n_obj - 1.0, 1.0)

    effective_w_features = params["w_features"] + ctx.valence * params["w_valence"]
    prior = softmax_prior(
        params["w_contrast"] * contrast
        + effective_w_features * ctx.feature_count
    )

    # Feature-proportional lexical competition (competition rate fixed at 0.5 per extra feature):
    # A speaker considering object r retrieves each of its true features with weight 1 / (1 + 0.5 * (fc - 1)),
    # and considers a false feature only with residual slip weight 0.025 / n_obj.
    fc_excess = jnp.maximum(ctx.feature_count - 1.0, 0.0)[None, :]
    true_avail = 1.0 / (1.0 + 0.5 * fc_excess)
    false_avail = 0.025 / n_obj
    real_avail = ctx.lex * true_avail + (1.0 - ctx.lex) * false_avail
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * real_avail

    heard = L2(params["alpha"], avail, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
