"""Pragmatic listener and speaker with prototype-extremity and uniqueness salience.

Listeners and speakers evaluate referents through their complexity extremity (squared
deviation from the scene's mean feature complexity) and contextual uniqueness (inverse
duplicate count across the display). On prior trials without an informative word,
listeners favor both complexity outliers (minimal/empty and maximal/full feature profiles)
and unique singletons over intermediate or duplicated objects; on heard-word trials,
the top-level pragmatic listener integrates this prior while simulating a depth-1
pragmatic speaker grounded in unbiased truth-conditional semantics.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_unique": dist.Normal(0.0, 1.0),
    "w_extremity": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
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
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_salience(ctx):
    real_mask = 1.0 - ctx.is_sink
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * real_mask[:, None, None], axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    total_dist = pair_dist + gray_diff + fam_diff
    clone_count = jnp.sum(total_dist <= EPS, axis=1)
    uniqueness = 1.0 / jnp.maximum(clone_count, 1.0)

    n_real = jnp.maximum(jnp.sum(real_mask), 1.0)
    fc_norm = ctx.feature_count / n_real
    fc_dev = fc_norm - jnp.mean(fc_norm)
    extremity = fc_dev * fc_dev
    return uniqueness, extremity


def listener(params, ctx):
    uniqueness, extremity = compute_salience(ctx)
    prior = softmax_prior(
        params["w_unique"] * uniqueness + params["w_extremity"] * extremity
    )
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
