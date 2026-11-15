"""Pragmatic listener at depth 2 with a perceptual crowding and feature-impurity object prior.

Before hearing an utterance, listeners and speakers share an object prior that penalizes
referents by their perceptual crowding (summed similarity to other objects in the scene)
and their internal feature impurity (the Bernoulli variance p * (1 - p) across the scene's
features, which vanishes for empty and all-feature objects and is smaller for minimal
single-feature items than for mixed intermediate items). At recursion depth 2, the pragmatic
listener inverts a speaker who plans for a depth-1 listener under this shared prior,
grounded in a literal listener with unbiased truth-conditional semantics.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_salience": dist.Normal(0.0, 2.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_prior(ctx, w_salience):
    real_mask = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_mask[:, None]
    n_feat = jnp.maximum(jnp.sum(real_mask), 1.0)
    n_obj = ctx.lex.shape[1]

    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-pair_dist) * (1.0 - eye)
    crowding = jnp.sum(sim, axis=1)

    p_feat = ctx.feature_count / n_feat
    impurity = p_feat * (1.0 - p_feat)

    return softmax_prior(-w_salience * (crowding + impurity))


def listener(params, ctx):
    prior = compute_prior(ctx, params["w_salience"])
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
