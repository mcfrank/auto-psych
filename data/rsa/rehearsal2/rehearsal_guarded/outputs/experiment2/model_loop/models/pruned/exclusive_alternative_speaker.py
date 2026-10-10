"""Exclusive alternative speaker and pragmatic listener.

Speakers formulate referring expressions by prioritizing dedicated exclusive
descriptors that uniquely single out the target, penalizing ambiguous descriptors
whenever the intended referent possesses a foolproof exclusive alternative in the
display. Pragmatic listeners invert this speaker by directing ambiguous expressions
toward candidate referents that lack any exclusive descriptor in the context.
On uninformative trials with no descriptive word, choices default to common-ground
visual singleton salience. A lapse parameter captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_exclusive": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[
    u: UTT,
    r: OBJ,
](
    alpha,
    w_exclusive,
    lex: ...,
    prior: ...,
    penalty: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - w_exclusive * at(penalty, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_exclusive_penalty(ctx):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]

    n_obj_per_feat = jnp.sum(real_lex, axis=1)
    is_exclusive_feat = jnp.where(n_obj_per_feat == 1, 1.0, 0.0) * real_words

    obj_has_exclusive = jnp.sum(real_lex * is_exclusive_feat[:, None], axis=0)
    has_exclusive = jnp.where(obj_has_exclusive > 0, 1.0, 0.0)

    penalty = has_exclusive[None, :] * (1.0 - is_exclusive_feat[:, None]) * real_lex
    return penalty


def compute_singleton_indicator(ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    return jnp.where(duplicate_count == 1, 1.0, 0.0)


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    penalty = compute_exclusive_penalty(ctx)

    heard = L1(
        params["alpha"],
        params["w_exclusive"],
        ctx.lex,
        prior,
        penalty,
    )[ctx.utterance]

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
