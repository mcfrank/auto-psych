"""Pragmatic listener reasoning about a speaker penalized for avoidable ambiguity.

When an object possesses a uniquely identifying expression that distinguishes it from
all context distractors, a cooperative speaker incurs an ambiguity penalty for choosing
a shared, non-identifying descriptor instead. When an object lacks any uniquely
identifying expression and cannot be uniquely named, the speaker incurs no penalty and
is forced to rely on shared descriptors. The pragmatic listener inverts this speaker,
inferring that ambiguous words signal referents that could not be uniquely named, while
relying on visual singleton salience on uninformative prior trials.
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
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., ambiguity_cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * (
                    log(L0[u, r](lex) + {EPS})
                    - at(ambiguity_cost, u, r)
                )
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_ambiguity_cost(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    real_words = (1.0 - is_sink)[:, None]
    ext = jnp.sum(lex * real_words, axis=1, keepdims=True)
    is_unique_word = ((ext == 1) & (real_words > 0)).astype(jnp.float32)
    can_uniquely_name = jnp.max(lex * is_unique_word, axis=0, keepdims=True)
    is_ambiguous_word = ((ext > 1) & (real_words > 0)).astype(jnp.float32)
    cost = (can_uniquely_name * is_ambiguous_word * lex).astype(jnp.float32)
    return cost


def compute_singleton(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    real_lex = lex * (1.0 - is_sink)[:, None]
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    is_identical = (pair_dist == 0).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_singleton = jnp.where(copy_count == 1, 1.0, 0.0)
    return is_singleton


def listener(params, ctx):
    cost = compute_ambiguity_cost(ctx.lex, ctx.is_sink)
    is_singleton = compute_singleton(ctx.lex, ctx.is_sink)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    heard = L1(
        params["alpha"],
        ctx.lex,
        prior,
        cost,
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
