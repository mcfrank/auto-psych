"""Speaker abstention listener: pragmatic listener reasoning about a speaker with an outside option.

Speakers in reference games consider an outside option to abstain from speaking
when all available expressions for an intended referent are ambiguous, preferring
silence over producing an uninformative referring expression. Pragmatic listeners
invert this choice: knowing that an ambiguous word would frequently have been
replaced by silence had the speaker intended an object lacking distinguishing features,
listeners infer that hearing an ambiguous word is less diagnostic of sparse targets.
This naturally moderates excessive scalar implicatures on two-object displays while
elevating choices of feature-rich objects on ubiquitous utterance displays.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "logit_p_abstain": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., speaker_prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(speaker_prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., speaker_prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(speaker_prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, speaker_prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_prior(ctx, params):
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

    return softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
    )


def compute_has_unique(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    match_counts = jnp.sum(real_lex, axis=1)
    is_unique_word = (match_counts == 1.0).astype(jnp.float32) * (
        1.0 - ctx.is_sink
    )
    obj_has_unique = jnp.max(real_lex * is_unique_word[:, None], axis=0)
    return obj_has_unique


def choice_probs(params, ctx):
    prior = compute_prior(ctx, params)
    has_unique = compute_has_unique(ctx)
    p_abstain = jax.nn.sigmoid(params["logit_p_abstain"]) * (1.0 - has_unique)
    p_speak = 1.0 - p_abstain

    speaker_prior = prior * p_speak + 1e-6
    speaker_prior = speaker_prior / jnp.sum(speaker_prior)

    heard = L2(params["alpha"], ctx.lex, speaker_prior)[ctx.utterance]
    probs = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(probs, params["lapse"])
