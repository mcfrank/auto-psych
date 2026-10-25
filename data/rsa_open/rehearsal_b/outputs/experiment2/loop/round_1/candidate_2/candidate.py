"""Type-level reference: reasoning over perceptual equivalence classes."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., inv_dup: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(inv_dup, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., inv_dup: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex, inv_dup) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_duplicates(ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    dup_count = jnp.sum(total_dist == 0, axis=1)
    is_singleton = jnp.where(dup_count == 1, 1.0, 0.0)
    inv_dup = 1.0 / dup_count
    return is_singleton, inv_dup


def listener(params, ctx):
    is_singleton, inv_dup = compute_duplicates(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    heard = L1(params["alpha"], ctx.lex, inv_dup, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
