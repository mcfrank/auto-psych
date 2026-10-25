"""Pragmatic listener reasoning about a speaker who penalizes avoidable ambiguity.

Communicative speakers select referring expressions by penalizing avoidable
ambiguity: candidate words are discounted proportionally to the excess competitors
they fail to eliminate compared to the most specific alternative available for that
referent. When an object cannot be uniquely named, the speaker communicates without
penalty using the most informative description available. Pragmatic listeners invert
this ambiguity-sensitive speaker, attributing underspecified words to referents that
lack more specific alternatives, while spontaneous visual attention on uninformative
displays defaults to contextually unique singletons.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ..., excess: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - beta * at(excess, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
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


def compute_excess(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    ext = jnp.sum(real_lex, axis=1)

    obj_word_ext = jnp.where(real_lex > 0, ext[:, None], 999.0)
    min_ext = jnp.min(obj_word_ext, axis=0)
    has_real_words = jnp.any(real_lex > 0, axis=0)
    min_ext = jnp.where(has_real_words, min_ext, 1.0)

    excess = jnp.where(
        real_lex > 0,
        jnp.maximum(0.0, ext[:, None] - min_ext[None, :]),
        0.0,
    )
    return excess


def listener(params, ctx):
    is_singleton = compute_singleton(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    excess = compute_excess(ctx)
    heard = L1(
        params["alpha"], params["beta"], ctx.lex, excess, prior
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
