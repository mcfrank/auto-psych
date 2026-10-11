"""Pragmatic listener reasoning about a trembling-hand speaker with speech production errors.

Listeners interpret referring expressions by modeling a trembling-hand speaker who occasionally
produces unintended utterances due to speech production noise. When a heard word is ambiguous,
the rational utility of speaking it intentionally is low, increasing the posterior probability
that the utterance was an unintended slip referring to a non-matching distractor. Choices on
uninformative trials are guided by discrete singleton uniqueness, contextual distinctiveness,
and visual feature complexity, with background motor lapses capturing random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "tremble": dist.Beta(1.0, 19.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_perceptual_features(ctx) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Discrete singleton indicator and continuous contextual distinctiveness."""
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diffs = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diffs, axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)
    distinct = jnp.sum(diffs, axis=(0, 2))
    return is_singleton, distinct


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
def S2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L2[u: UTT, r: OBJ](joint: ...):
    listener: thinks[
        speaker: chooses(r in OBJ, u in UTT, wpp=at(joint, u, r)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    is_singleton, distinct = compute_perceptual_features(ctx)
    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
    )

    s2 = S2(params["alpha"], ctx.lex, prior)
    real_words = (1.0 - ctx.is_sink)[:, None]
    n_real = jnp.sum(1.0 - ctx.is_sink)
    tremble_dist = real_words / jnp.maximum(n_real, 1.0)

    s_policy = (1.0 - params["tremble"]) * s2 + params["tremble"] * tremble_dist
    joint = s_policy * prior[None, :]

    heard = L2(joint)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
