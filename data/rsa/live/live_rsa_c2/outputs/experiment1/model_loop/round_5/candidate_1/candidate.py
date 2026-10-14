"""RSA depth-2 pragmatic listener reasoning about an ambiguity-escaping speaker.

When communicating about an object that cannot be uniquely named in context because
every applicable word is ambiguous, speakers frequently take an outside option
(withholding speech or using an unmodelled descriptive gesture) rather than uttering
an unreliably ambiguous label. Conversely, when an intended referent possesses an
exclusive distinguishing feature that permits unique identification, the speaker
communicates normally. Pragmatic listeners invert this ambiguity-escaping speaker
at depth 2: because a speaker intending an unnameable, minimally featured target
frequently takes the outside option, hearing an ambiguous label provides weakened
evidence for the minimal target, naturally elevating choices for competitor referents
that possess distinguishing features or redundant attributes.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

ACTIONS = jnp.arange(2)

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_escape": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_empty": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_unnameability(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Compute referential unnameability for each object in context."""
    real_lex = lex * (1.0 - is_sink)[:, None]
    col_sums = jnp.sum(real_lex, axis=1, keepdims=True)
    l0 = jnp.where(col_sums > 0, real_lex / (col_sums + 1e-10), 0.0)
    sink_lex = lex * is_sink[:, None]
    sink_sums = jnp.sum(sink_lex, axis=1, keepdims=True)
    sink_l0 = jnp.where(sink_sums > 0, sink_lex / (sink_sums + 1e-10), 0.0)
    total_l0 = l0 + sink_l0
    max_inform = jnp.max(total_l0, axis=0)
    return jnp.maximum(1.0 - max_inform, 0.0)


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def S1[u: UTT, r: OBJ](alpha, w_escape, lex: ..., unnameability: ...):
    speaker: knows(r)
    speaker: chooses(
        act in ACTIONS,
        cand in UTT,
        wpp=(1 - act) * at(lex, cand, r) * exp(alpha * log(L0[cand, r](lex) + {EPS}))
        + act * (cand == 0) * vec(unnameability, r) * exp(w_escape),
    )
    return Pr[speaker.act == 0, speaker.cand == u]


@memo
def L1[u: UTT, r: OBJ](lex: ..., prior: ..., s1: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * vec(prior, r) * at(s1, u, r),
    )
    return Pr[listener.r == r]


@memo
def S2[u: UTT, r: OBJ](alpha, w_escape, lex: ..., l1: ..., unnameability: ...):
    speaker: knows(r)
    speaker: chooses(
        act in ACTIONS,
        cand in UTT,
        wpp=(1 - act) * at(lex, cand, r) * exp(alpha * log(at(l1, cand, r) + {EPS}))
        + act * (cand == 0) * vec(unnameability, r) * exp(w_escape),
    )
    return Pr[speaker.act == 0, speaker.cand == u]


@memo
def L2[u: UTT, r: OBJ](lex: ..., prior: ..., s2: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * vec(prior, r) * at(s2, u, r),
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

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_empty"] * is_empty
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    unnameability = compute_unnameability(ctx.lex, ctx.is_sink)

    s1 = S1(params["alpha"], params["w_escape"], ctx.lex, unnameability)
    l1 = L1(ctx.lex, prior, s1)
    s2 = S2(params["alpha"], params["w_escape"], ctx.lex, l1, unnameability)
    heard = L2(ctx.lex, prior, s2)[ctx.utterance]

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
