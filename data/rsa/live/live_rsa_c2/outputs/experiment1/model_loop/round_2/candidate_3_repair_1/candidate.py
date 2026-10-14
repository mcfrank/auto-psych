"""RSA depth-2 pragmatic listener with word-to-feature lexical uncertainty.

When interpreting a referring expression, listeners reason with lexical uncertainty
about which feature in the context the speaker's word picks out. Rather than
rigidly binding the heard utterance to a single attribute with probability one,
the listener considers that the speaker's word denotes its matching feature with high
probability but may denote an alternative feature present in the visual scene with
a baseline uncertainty rate beta. Pragmatic speakers anticipate this lexical uncertainty,
and depth-2 listeners invert the resulting communicative distribution. Because candidate
referents with multiple distinguishing features match multiple contextual attributes,
feature uncertainty softens recursive preemption penalties against overspecified
competitors and naturally accounts for low-rate false-feature selections.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.Beta(1.0, 9.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., kernel: ...):
    listener: knows(u)
    listener: chooses(f in UTT, wpp=at(kernel, u, f))
    listener: chooses(r in OBJ, wpp=at(lex, f, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., kernel: ..., prior: ...):
    listener: knows(u)
    listener: chooses(f in UTT, wpp=at(kernel, u, f))
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, kernel) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is listener.f
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., kernel: ..., prior: ...):
    listener: knows(u)
    listener: chooses(f in UTT, wpp=at(kernel, u, f))
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, kernel, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is listener.f
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

    n_utt = ctx.lex.shape[0]
    eye = jnp.eye(n_utt)
    is_sink_col = ctx.is_sink[:, None]
    is_sink_row = ctx.is_sink[None, :]
    sink_mask = (is_sink_col > 0) | (is_sink_row > 0)

    # Real utterances have 1.0 on diagonal, beta off-diagonal, and 0 to/from sink.
    kernel = jnp.where(eye > 0, 1.0, params["beta"])
    kernel = jnp.where(sink_mask, eye, kernel)

    heard = L2(params["alpha"], ctx.lex, kernel, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
