"""Pragmatic listener with bounded-precision softmax decision rule and speaker transmission noise.

Refines softmax_belief_listener by incorporating stochastic transmission noise from
noisy_channel_speaker_l1. The speaker rationally selects an intended descriptor based on literal
informativeness, but articulates applicable features with stochastic execution noise.
This transmission noise softens extreme pragmatic likelihood ratios when competitors possess
multiple additional features or duplicate copies, correcting overpredictions on large feature gap
trials, while the downstream softmax decision rule translates subjective posterior beliefs into
bounded choice probabilities.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "epsilon": dist.Beta(1.0, 5.0),
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
def L1[u: UTT, r: OBJ](alpha, epsilon, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            m in UTT,
            wpp=at(lex, m, r) * exp(alpha * log(L0[m, r](lex) + {EPS})),
        ),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * ((1.0 - epsilon) * (u == m) + epsilon),
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

    heard = L1(
        params["alpha"], params["epsilon"], ctx.lex, prior
    )[ctx.utterance]

    matches = real_lex[ctx.utterance]
    utt_logits = params["beta"] * heard
    utt_weights = matches * jnp.exp(utt_logits - jnp.max(utt_logits))
    n_matches = jnp.sum(matches)
    safe_weights = jnp.where(n_matches > 0, utt_weights, 1.0)
    utt_choice = safe_weights / (jnp.sum(safe_weights) + EPS)

    prior_logits = params["beta"] * prior
    prior_weights = jnp.exp(prior_logits - jnp.max(prior_logits))
    prior_choice = prior_weights / (jnp.sum(prior_weights) + EPS)

    choice = jnp.where(ctx.is_prior > 0, prior_choice, utt_choice)
    return with_lapse(choice, params["lapse"])
