"""Pragmatic listener at depth 2 with soft semantic typicality, absence-oddball plain salience, and crowded-utterance literal grounding."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_contrast": dist.LogNormal(0.0, 0.5),
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


def listener(params, ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask
    n_obj = ctx.lex.shape[1]

    # Pairwise feature distance between objects and smooth nearest-neighbor perceptual contrast
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    denom = jnp.maximum(n_obj - 1.0, 1.0)
    contrast = jnp.sum(1.0 - jnp.exp(-pair_dist), axis=1) / denom

    # Plain (0-feature) objects suffer a baseline deficit in sparse scenes but pop out as an absence oddball
    # when surrounded by dense multi-feature competitors (mean pairwise feature distance > 2).
    is_plain = jnp.where(ctx.feature_count == 0, 1.0, 0.0)
    mean_dist = jnp.sum(pair_dist, axis=1) / denom
    salience = (
        contrast
        + 0.235 * (ctx.feature_count - 0.5 * is_plain)
        + 0.47 * is_plain * jnp.maximum(mean_dist - 2.0, 0.0)
    )
    prior = softmax_prior(params["w_contrast"] * salience)

    # Soft semantic typicality and extension-driven overextension:
    fc_excess = jnp.maximum(ctx.feature_count - 1.0, 0.0)[None, :]
    true_typ = 1.0 / (1.0 + 0.5 * fc_excess)

    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    overext = jnp.maximum(feat_ext - 2.0, 0.0)
    false_typ = (0.015 + 0.25 * overext) / n_obj

    real_sem = ctx.lex * true_typ + (1.0 - ctx.lex) * false_typ
    sem = ctx.is_sink[:, None] * ctx.lex + real_mask * real_sem

    prag = L2(params["alpha"], sem, prior)[ctx.utterance]

    # When the heard word is over-extended across >2 objects, referential crowding partially attenuates
    # depth-2 recursion toward literal grounding with residual foil leakage.
    heard_ext = jnp.sum(real_lex[ctx.utterance])
    heard_overext = jnp.maximum(heard_ext - 2.0, 0.0)
    lit_u = real_lex[ctx.utterance] + 0.004 * (1.0 - real_lex[ctx.utterance])
    lit = lit_u / jnp.maximum(jnp.sum(lit_u), EPS)
    ret_lit = 0.28 * heard_overext
    heard = (1.0 - ret_lit) * prag + ret_lit * lit

    return jnp.where(ctx.is_prior > 0, prior, heard)
