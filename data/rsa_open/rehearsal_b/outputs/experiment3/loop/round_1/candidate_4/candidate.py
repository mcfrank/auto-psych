"""Pragmatic listener reasoning under referential feature uncertainty with contextual distinctiveness.

Refining feature_uncertainty_listener_2 by incorporating continuous contextual
distinctiveness from rsa_l1_singleton_feat_color_valence_l0 into the visual
attention prior: listeners interpret referring expressions under referential
feature uncertainty over literal semantics, but visual attention weights
candidate referents by combining discrete singleton salience with continuous
feature contrast relative to display competitors. This directly accounts for
the contextual uniqueness of unmentioned features in scalar competitor displays
and scales singleton salience in large competitor sets, addressing empirical
discrepancies in competitor selection and singleton salience. Communicative
speakers select expressions anticipating this feature-attribution inference,
while spontaneous visual attention on uninformative displays follows the
combined perceptual prior.

Differences from feature_uncertainty_listener_2: recursion depth remains
depth 1 (listener calls L1); parameter w_distinct is added with prior
dist.Normal(0.0, 1.0); no parameters are removed (retaining alpha and
w_singleton); the object prior is extended by adding w_distinct times
contextual distinctiveness (summed pairwise Hamming distance across context
objects for real features); memo definitions (L0 and L1) remain unchanged;
and listener passes the extended prior into L1.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex_norm: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex_norm, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., lex_norm: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex_norm) + {EPS})),
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


def compute_distinct(ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    return jnp.sum(diff, axis=(0, 2))


def listener(params, ctx):
    is_singleton = compute_singleton(ctx)
    distinct = compute_distinct(ctx)
    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
    )

    fc = jnp.maximum(ctx.feature_count, 1.0)
    lex_norm = jnp.where(
        ctx.is_sink[:, None] > 0,
        ctx.lex,
        ctx.lex / fc[None, :],
    )

    heard = L1(params["alpha"], ctx.lex, lex_norm, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
