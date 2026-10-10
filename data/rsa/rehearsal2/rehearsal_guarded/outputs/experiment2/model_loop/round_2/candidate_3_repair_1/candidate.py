"""Pragmatic listener reasoning under contextual lexical alignment uncertainty.

Listeners interpret referring expressions under uncertainty about which visual feature a
word picks out, reasoning that an utterance can be associated with features that contextually
co-occur with the named property across objects in the display. When interpreting a word,
pragmatic listeners model a speaker who selects an informative feature to describe the
intended referent and transmits that feature through this contextual alignment mapping,
inverting the joint process to infer the referent. On uninformative trials with no descriptive
word, choices default to common-ground visual singleton salience.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "logit_theta": dist.Normal(-3.5, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
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
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., trans: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            f in UTT,
            wpp=at(lex, f, r)
            * exp(alpha * log(L1[f, r](alpha, lex, prior) + {EPS})),
        ),
        speaker: chooses(u in UTT, wpp=at(trans, u, f)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


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


def compute_trans(ctx, theta):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    C = real_lex @ real_lex.T
    counts = jnp.sum(real_lex, axis=1)
    union = counts[:, None] + counts[None, :] - C
    denom = jnp.maximum(union, 1.0)
    jaccard = C / denom

    n_utt = ctx.is_sink.shape[0]
    eye = jnp.eye(n_utt)
    real_trans = (
        jnp.where(eye > 0, 1.0 - theta, theta * jaccard)
        * real_words[:, None]
        * real_words[None, :]
    )
    sink_trans = jnp.outer(ctx.is_sink, ctx.is_sink)
    return real_trans + sink_trans


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    theta = jax.nn.sigmoid(params["logit_theta"])
    trans = compute_trans(ctx, theta)

    heard = L2(params["alpha"], ctx.lex, prior, trans)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
