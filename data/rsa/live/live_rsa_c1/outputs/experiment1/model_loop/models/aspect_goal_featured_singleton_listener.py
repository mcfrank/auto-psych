"""Pragmatic listener inverting an aspect-goal speaker with feature-gated singleton salience.

Listeners recognize that speakers alternate between referent identification and aspect description.
In foundational scene parsing, visual singleton salience requires the presence of visual features:
an empty object lacking features does not pre-attentively pop out merely because it lacks duplicates.
Refining speaker_aspect_goal_listener by making visual singleton salience contingent on visual feature
presence, preventing the over-selection of empty singletons in displays with duplicated full objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "p_aspect": dist.Beta(1.0, 5.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}

GOALS = (0, 1)


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, p_aspect, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            g in GOALS,
            wpp=(1.0 - p_aspect) if g == 0 else p_aspect,
        ),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * ((1.0 - g) * exp(alpha * log(L0[u, r](lex) + {EPS})) + g * 1.0),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    duplicate_count = jnp.sum(pair_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)
    featured_singleton = is_singleton * jnp.where(ctx.feature_count > 0, 1.0, 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    prior = softmax_prior(
        params["w_singleton"] * featured_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
    )

    heard = L1(params["alpha"], params["p_aspect"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
