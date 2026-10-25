"""Sortal partition pragmatic listener.

People interpret and produce referring expressions over sortal equivalence classes
of objects rather than individual visual tokens. Communicative speakers choose words
expecting listeners to identify referent types, while pragmatic listeners partition
hypothesis weight across duplicate tokens within each sortal class, preventing
identical distractors from spuriously diluting unique minimal referents. On
uninformative displays without an informative word, visual attention defaults to
contextually unique singletons.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 0.5),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., cluster_weight: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * vec(cluster_weight, r),
    )
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., cluster_weight: ..., token_prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(token_prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex, cluster_weight) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_cluster_and_singleton(ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)
    return duplicate_count.astype(jnp.float32), is_singleton


def listener(params, ctx):
    cluster_size, is_singleton = compute_cluster_and_singleton(ctx)
    cluster_weight = cluster_size ** (-params["gamma"])
    token_prior = softmax_prior(
        params["w_singleton"] * is_singleton
        - params["gamma"] * jnp.log(cluster_size)
    )
    heard = L1(
        params["alpha"], ctx.lex, cluster_weight, token_prior
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, token_prior, heard)
