"""Pragmatic listener reasoning under referential feature uncertainty with communicative feature parsimony.

Refining feature_uncertainty_listener_2 by incorporating communicative feature
parsimony from bounded_depth_feature_parsimony: listeners interpret referring
expressions under referential feature uncertainty, weighting literal semantics
by the probability that the heard feature was selected from an object's feature
bundle, but additionally discount candidate referents possessing redundant visual
features through communicative feature parsimony. On informative trials, this
parsimony penalty sharpens rejection of intermediate competitors in multi-level
scalar hierarchies and prevents visual singleton salience from spuriously
elevating cluttered candidates, while spontaneous visual attention on uninformative
displays defaults to contextually unique singletons.

Differences from feature_uncertainty_listener_2: recursion depth remains depth 1
(listener calls L1); parameter beta is added with prior dist.LogNormal(0.0, 1.0);
no parameters are removed (retaining alpha and w_singleton); L1 adds beta and
feature_count to its signature and scales referent choice weights by
exp(-beta * vec(feature_count, r)); and listener passes beta and
ctx.feature_count into L1 on informative trials.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex_norm: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex_norm, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ..., lex_norm: ..., prior: ..., feature_count: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex_norm) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r)
        * Pr[speaker.r == r]
        * exp(-beta * vec(feature_count, r)),
    )
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
    prior = softmax_prior(params["w_singleton"] * is_singleton)

    fc = jnp.maximum(ctx.feature_count, 1.0)
    lex_norm = jnp.where(
        ctx.is_sink[:, None] > 0,
        ctx.lex,
        ctx.lex / fc[None, :],
    )

    heard = L1(
        params["alpha"],
        params["beta"],
        ctx.lex,
        lex_norm,
        prior,
        ctx.feature_count,
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
