"""Pragmatic listener reasoning under lexical uncertainty over semantic specificity.

Listeners interpret referring expressions under lexical uncertainty, being unsure
whether the speaker intends a permissive semantic reading true of any object
possessing the feature or a specific reading true only of minimal exemplars.
Communicative speakers select expressions by anticipating this semantic uncertainty
across both candidate lexica. On uninformative displays without an informative word,
visual attention defaults to contextually unique singletons.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex_perm: ..., lex_strict: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=0.5
            * at(lex_perm, u, r)
            * exp(alpha * log(L0[u, r](lex_perm) + {EPS}))
            + 0.5
            * at(lex_strict, u, r)
            * exp(alpha * log(L0[u, r](lex_strict) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=vec(prior, r) * Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_singleton(ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    return jnp.where(duplicate_count == 1, 1.0, 0.0)


def compute_lex_strict(ctx):
    is_real = (1.0 - ctx.is_sink)[:, None]
    fc = ctx.feature_count[None, :]
    masked_fc = jnp.where((ctx.lex > 0) & (is_real > 0), fc, 999.0)
    min_fc = jnp.min(masked_fc, axis=1, keepdims=True)
    is_strict = jnp.where(
        is_real > 0,
        jnp.where((ctx.lex > 0) & (fc == min_fc), 1.0, 0.0),
        ctx.lex,
    )
    return is_strict


def listener(params, ctx):
    is_singleton = compute_singleton(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    lex_strict = compute_lex_strict(ctx)
    heard = L1(params["alpha"], ctx.lex, lex_strict, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
