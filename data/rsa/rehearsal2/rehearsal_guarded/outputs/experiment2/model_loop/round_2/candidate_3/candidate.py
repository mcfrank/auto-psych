"""Pragmatic listener reasoning under contextual lexical alignment uncertainty.

Listeners interpret referring expressions under lexical uncertainty about which
visual feature a word picks out, grounded in contextual feature co-occurrence across
scene objects. Rather than assuming words map deterministically to features in isolation,
listeners allocate semantic interpretation probability across features that co-occur
with the named attribute in the display. Pragmatic listeners invert speakers who
evaluate communicative recovery under this soft, co-occurrence-based lexical alignment,
while defaulting to visual singleton salience on uninformative displays.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "kappa": dist.Beta(1.0, 3.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., align: ...):
    listener: knows(u)
    listener: chooses(f in UTT, wpp=at(align, u, f))
    listener: chooses(r in OBJ, wpp=at(lex, f, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., align: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, align) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., align: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, align, prior) + {EPS})),
        ),
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


def compute_alignment(ctx, kappa):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    C = real_lex @ real_lex.T
    denom = jnp.maximum(jnp.diag(C)[:, None], 1.0)
    overlap = C / denom

    n_utt = ctx.is_sink.shape[0]
    eye = jnp.eye(n_utt)
    real_align = (
        jnp.where(eye > 0, 1.0, kappa * overlap)
        * real_words[None, :]
        * real_words[:, None]
    )
    sink_align = jnp.outer(ctx.is_sink, ctx.is_sink)
    return real_align + sink_align


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    align = compute_alignment(ctx, params["kappa"])

    heard = L2(params["alpha"], ctx.lex, align, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
