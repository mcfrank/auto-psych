"""Bounded reasoning depth pragmatic listener with visual feature parsimony.

Refining bounded_depth_listener by incorporating visual feature parsimony into the
referent prior: listeners interpret referring expressions with bounded reasoning depth
(depth 1), inverting an informative speaker who evaluates literal semantics, while their
referent prior integrates visual singleton salience with a parsimony penalty against objects
with superfluous features. On prior trials, this assigns greater spontaneous belief to empty
zero-feature singletons over partially featured items; on informative trials, this prevents
spurious singleton bonuses from overriding scalar implicature, reducing over-allocation to
multi-feature singleton competitors when words match duplicate minimal targets.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
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


def listener(params, ctx):
    is_singleton = compute_singleton(ctx)
    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_features"] * ctx.feature_count
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
