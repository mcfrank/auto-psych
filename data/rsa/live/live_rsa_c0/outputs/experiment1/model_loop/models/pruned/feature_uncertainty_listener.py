"""Listeners interpret referring expressions under uncertainty about which visual feature a speaker's word picks out.

Rather than assuming an error-free mapping from words to visual features,
pragmatic listeners reason that a speaker selects an informative descriptive feature
of the intended referent but transmits an expression subject to feature confusion
among available contextual alternatives. This feature uncertainty naturally produces
non-zero selection of feature-rich foils in complex displays, moderates overly severe
scalar exclusion, and maintains literal accuracy on simple unambiguous displays.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "logit_omega": dist.Normal(-3.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., channel: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            f in UTT,
            wpp=at(lex, f, r) * exp(alpha * log(L0[f, r](lex) + {EPS})),
        ),
        speaker: chooses(u in UTT, wpp=at(channel, f, u)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., channel: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            f in UTT,
            wpp=at(lex, f, r)
            * exp(
                alpha * log(L1[f, r](alpha, lex, prior, channel) + {EPS})
            ),
        ),
        speaker: chooses(u in UTT, wpp=at(channel, f, u)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_channel(is_sink: jnp.ndarray, omega) -> jnp.ndarray:
    real_mask = 1.0 - is_sink
    n_real = jnp.sum(real_mask)
    n_utt = is_sink.shape[0]
    eye = jnp.eye(n_utt)

    denom_real = jnp.maximum(1.0, n_real - 1.0)
    real_to_other = (
        (real_mask[:, None] * real_mask[None, :])
        * (1.0 - eye)
        * (omega / denom_real)
    )
    real_to_same = (
        (real_mask[:, None] * real_mask[None, :]) * eye * (1.0 - omega)
    )

    denom_sink = jnp.maximum(1.0, n_real)
    sink_to_real = (is_sink[:, None] * real_mask[None, :]) * (
        omega / denom_sink
    )
    sink_to_sink = (is_sink[:, None] * is_sink[None, :]) * (1.0 - omega)

    channel = real_to_other + real_to_same + sink_to_real + sink_to_sink
    return jnp.where(
        n_real == 1.0,
        jnp.where(real_mask[:, None] * eye > 0, 1.0, channel),
        channel,
    )


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

    effective_w_features = (
        params["w_features"] + ctx.valence * params["w_valence"]
    )

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + effective_w_features * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    omega = jax.nn.sigmoid(params["logit_omega"])
    channel = compute_channel(ctx.is_sink, omega)

    heard = L2(params["alpha"], ctx.lex, prior, channel)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
