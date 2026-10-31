"""Cognitive-hierarchy Theory of Mind with collateral unmentioned-feature interference and Weber-scaled perceptual salience."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_salience": dist.LogNormal(0.0, 0.5),
}


@memo
def L0[u: UTT, r: OBJ](avail: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(avail, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, avail: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(avail, u, r) * exp(alpha * log(L0[u, r](avail) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, avail: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(avail, u, r)
            * exp(alpha * log(0.85 * L1[u, r](alpha, avail, prior) + 0.15 * L0[u, r](avail) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask
    n_obj = ctx.lex.shape[1]
    n_comp = jnp.maximum(n_obj - 1.0, 1.0)

    # Weber-scaled perceptual contrast across competitors (including color and familiarization differences),
    # with absence-oddball pop-out for featureless objects surrounded by dense competitors.
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = (
        jnp.sum(diff, axis=0)
        + jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
        + jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    )
    mean_dist = jnp.sum(pair_dist, axis=1) / n_comp
    plain_gate = 1.5 / n_comp + 0.55 * jnp.maximum(mean_dist - 2.0, 0.0)
    pos_gate = jnp.where(ctx.feature_count == 0, plain_gate, 1.0 - 0.25 * jnp.exp(-ctx.feature_count))
    contrast = (0.88 / jnp.sqrt(n_comp)) * pos_gate * jnp.sum(1.0 - jnp.exp(-pair_dist), axis=1)

    salience = (
        contrast
        + (0.22 + 0.50 * ctx.valence) * ctx.feature_count
        + 0.30 * (1.0 - ctx.grayscale)
        + 0.60 * ctx.familiarization
    )
    prior = softmax_prior(params["w_salience"] * salience)

    # Collateral lexical interference from unmentioned diagnostic features on the target referent:
    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    w_sel = jnp.clip((n_obj - feat_ext) / jnp.maximum(n_obj - 2.0, 1.0), 0.0, 1.0) * real_mask
    obj_load = jnp.sum(real_lex * w_sel, axis=0, keepdims=True)
    rival_load = jnp.maximum(obj_load - real_lex * w_sel, 0.0)
    true_avail = 1.0 / (1.0 + 0.22 * rival_load)
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * (ctx.lex * true_avail)

    prag = L2(params["alpha"], avail, prior)[ctx.utterance]
    lit0 = L0(avail)[ctx.utterance]

    # Epistemic horizon weights gated by utterance referential fan-out in crowded (>=4 object) scenes:
    heard_ext = jnp.sum(real_lex[ctx.utterance])
    overext = jnp.maximum(heard_ext - 2.0, 0.0) * jnp.clip(n_obj - 3.0, 0.0, 1.0)
    w_l0 = 0.32 * overext
    w_prior = 0.0065 * overext
    heard = (1.0 - w_l0 - w_prior) * prag + w_l0 * lit0 + w_prior * prior

    return jnp.where(ctx.is_prior > 0, prior, heard)
