"""Pragmatic listener inverting a nested-choice speaker with substitute descriptors.

When selecting referring expressions, speakers face a hierarchical decision process:
descriptors that pick out the exact same extension of objects in the scene form an equivalence
class of functional substitutes. Rather than treating each redundant descriptor as an independent
competitor that multiplies the probability of choosing that category, a nested-choice speaker
discounts substitute descriptors according to co-extensional class size. Pragmatic listeners invert
this nested speaker, preventing the severe over-penalization of multi-featured competitors when
hearing a shared descriptor. On uninformative mumble trials, choices reflect perceptual and
experiential priors over object features.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "nu": dist.Beta(1.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
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
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., sub_weight: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * vec(sub_weight, u)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
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

    diff_u = jnp.abs(real_lex[:, None, :] - real_lex[None, :, :])
    is_coext = jnp.all(diff_u == 0, axis=-1)
    non_sink = 1.0 - ctx.is_sink
    is_coext = is_coext * non_sink[:, None] * non_sink[None, :]
    coext_count = jnp.sum(is_coext, axis=1) + ctx.is_sink
    sub_weight = jnp.power(coext_count, -params["nu"])

    heard = L1(params["alpha"], ctx.lex, prior, sub_weight)[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
