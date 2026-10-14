"""Pragmatic listener inverting a speaker under contextual graded semantic applicability.

Listeners interpret utterances under graded semantic applicability: words do not function as rigid
boolean filters, but have applicability that decays with an object's excess unmentioned visual
features relative to the visual scene's minimal matching exemplar. Foundational literal interpretation
favors the minimal, focal exemplar that possesses no extraneous features, while penalizing complex
and intermediate competitors in proportion to their descriptive dilution. A rational pragmatic
listener inverts a communicative speaker operating over this graded lexicon, directly suppressing
intermediate competitor choices and elevating minimal-feature target selection on ambiguous and
redundant-feature trials. On uninformative mumble trials, choices directly reflect perceptual scene
expectations governed by feature-gated singleton pop-out, visual complexity extremity, and total feature count.
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
    "w_extremity": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
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
    featured_singleton = is_singleton * (ctx.feature_count > 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    min_count = jnp.min(ctx.feature_count)
    max_count = jnp.max(ctx.feature_count)
    midpoint = 0.5 * (min_count + max_count)
    extremity = jnp.abs(ctx.feature_count - midpoint)

    prior = softmax_prior(
        params["w_singleton"] * featured_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_extremity"] * extremity
        + params["w_color"] * (1.0 - ctx.grayscale)
        + params["w_familiar"] * ctx.familiarization
    )

    # Graded semantic applicability:
    # For each utterance, compute excess features relative to the minimal matching exemplar
    counts = jnp.where(ctx.lex > 0.0, ctx.feature_count[None, :], 999.0)
    min_counts = jnp.min(counts, axis=1, keepdims=True)
    excess = jnp.maximum(ctx.feature_count[None, :] - min_counts, 0.0)
    real_graded = ctx.lex * jnp.exp(-params["gamma"] * excess)
    graded_lex = jnp.where(ctx.is_sink[:, None] > 0.0, ctx.lex, real_graded)

    heard = L1(params["alpha"], graded_lex, prior)[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
