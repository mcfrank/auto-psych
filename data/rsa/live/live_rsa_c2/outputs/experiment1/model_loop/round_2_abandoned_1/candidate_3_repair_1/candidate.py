"""RSA pragmatic listener at depth 2 under feature-reference uncertainty.

Listeners interpret referring expressions under feature-reference uncertainty:
a speaker's word primarily denotes its nominal visual feature, but the listener
maintains residual uncertainty about which feature in the display was intended.
When resolving reference, listeners marginalize over candidate visual features,
elevating the communicative viability of overspecified competitor objects that
possess multiple candidate attributes. On uninformative trials without an informative
word, choices follow a perceptual referent prior combining singleton uniqueness,
contextual distinctiveness, visual feature complexity, and color contrast.
A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "theta": dist.Beta(1.0, 9.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
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


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
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
    )

    l2_table = L2(params["alpha"], ctx.lex, prior)

    # Feature-reference uncertainty:
    # When resolving heard word u, the listener assigns probability (1 - theta)
    # to intended feature u and distributes theta across all real features in the context.
    real_mask = 1.0 - ctx.is_sink
    n_real = jnp.maximum(jnp.sum(real_mask), 1.0)
    theta = params["theta"]

    u_one_hot = jnp.eye(ctx.lex.shape[0])[ctx.utterance]
    feature_weights = (
        (1.0 - theta) * u_one_hot + theta * (real_mask / n_real)
    ) * real_mask
    feature_weights = feature_weights / jnp.maximum(jnp.sum(feature_weights), 1e-10)

    heard = jnp.dot(feature_weights, l2_table)

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
