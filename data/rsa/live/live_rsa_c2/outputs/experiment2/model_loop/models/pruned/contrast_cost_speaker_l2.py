"""RSA depth-2 pragmatic listener reasoning about a speaker with cognitive contrast verification costs.

When formulating referring expressions in visual reference games, speakers balance
communicative informativeness against the cognitive verification cost of using
distinctive attributes: deploying an exclusive or distinguishing label requires
visually searching and verifying that context distractors do not possess that
attribute. This verification effort scales with the utterance's contextual
discriminative contrast (the proportion of context distractors ruled out), prompting
speakers to frequently produce accessible, shared category labels even when an
exclusive distinguishing feature is available. Pragmatic listeners invert this
verification-bounded speaker via recursive Theory of Mind, naturally softening
counterfactual preemption in subset displays and elevating simpler targets in
cross-cutting contexts.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "kappa": dist.LogNormal(0.0, 1.0),
    "w_contrast_cost": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_empty": dist.Normal(0.0, 1.0),
    "w_endpoint": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def S1[u: UTT, r: OBJ](alpha, alt_weight: ..., lex: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r)
        * vec(alt_weight, u)
        * exp(alpha * log(L0[u, r](lex) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L1[u: UTT, r: OBJ](kappa, lex: ..., prior: ..., s1: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * vec(prior, r)
        * exp(kappa * log(at(s1, u, r) + {EPS})),
    )
    return Pr[listener.r == r]


@memo
def S2[u: UTT, r: OBJ](alpha, alt_weight: ..., lex: ..., l1: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r)
        * vec(alt_weight, u)
        * exp(alpha * log(at(l1, u, r) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L2[u: UTT, r: OBJ](kappa, lex: ..., prior: ..., s2: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * vec(prior, r)
        * exp(kappa * log(at(s2, u, r) + {EPS})),
    )
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

    is_empty = jnp.where(ctx.feature_count == 0, 1.0, 0.0)

    k_min = jnp.min(ctx.feature_count)
    k_max = jnp.max(ctx.feature_count)
    is_inter = jnp.where(
        (ctx.feature_count > k_min) & (ctx.feature_count < k_max), 1.0, 0.0
    )
    is_endpoint = 1.0 - is_inter

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_empty"] * is_empty
        + params["w_endpoint"] * is_endpoint
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    n_obj = ctx.lex.shape[1]
    freq = jnp.sum(real_lex, axis=1)
    # Contextual discriminative contrast: proportion of distractors eliminated by u
    # Scale from 0 (true of all items) to 1 (unique identifying feature)
    contrast = (n_obj - freq) / jnp.maximum(n_obj - 1.0, 1.0)
    contrast = jnp.where(ctx.is_sink > 0, 0.0, contrast)
    alt_weight = jnp.exp(-params["w_contrast_cost"] * contrast)

    s1 = S1(params["alpha"], alt_weight, ctx.lex)
    l1 = L1(params["kappa"], ctx.lex, prior, s1)
    s2 = S2(params["alpha"], alt_weight, ctx.lex, l1)
    heard = L2(params["kappa"], ctx.lex, prior, s2)[ctx.utterance]

    p_choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(p_choice, params["lapse"])
