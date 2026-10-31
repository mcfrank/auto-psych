"""Pragmatic listener at depth 2 reasoning about an aspect-oriented question under discussion (QUD)."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_salience": dist.LogNormal(0.0, 0.5),
}


@memo
def L2[u: UTT, r: OBJ](alpha, avail: ..., u1: ..., qud_prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(qud_prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(avail, u, r) * exp(alpha * at(u1, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask
    n_obj = ctx.lex.shape[1]

    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)

    dup_count = jnp.sum(jnp.exp(-10.0 * pair_dist), axis=1)

    fc = ctx.feature_count
    shared_pos = 0.5 * (fc[:, None] + fc[None, :] - pair_dist)
    tversky_d = pair_dist / (1.0 + 0.35 * shared_pos)
    mean_dm = (
        0.5
        * (jnp.sum(pair_dist, axis=1) + jnp.sum(fc) - n_obj * fc)
        / jnp.maximum(n_obj - 1.0, 1.0)
    )
    oddball_absence = jnp.maximum(mean_dm - 2.0, 0.0)

    pos_gate = 1.0 - 0.30 * jnp.exp(-fc)
    contrast = (
        pos_gate
        * jnp.sum(1.0 - jnp.exp(-pair_dist) + 0.30 * tversky_d, axis=1)
        / jnp.maximum(n_obj - 1.0, 1.0)
        + 0.75 * oddball_absence
    )

    feat_val = 0.5 * fc + 0.5 * jnp.log1p(fc)
    salience = contrast + (0.68 + 0.34 * ctx.valence) * feat_val

    fc_pos = jnp.maximum(fc, 1.0)
    qud_mult = jnp.maximum(fc_pos, dup_count)
    qud_log_mod = 0.52 * jnp.log(fc_pos / qud_mult) + 0.22 * jnp.log(fc_pos)

    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    overext = jnp.maximum(feat_ext - 2.0, 0.0)
    false_avail = (0.015 + 0.22 * overext) / n_obj
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * (
        ctx.lex + (1.0 - ctx.lex) * false_avail
    )
    l0 = avail / jnp.sum(avail, axis=1, keepdims=True)
    u0 = jnp.log(0.35 + 0.65 * l0 + EPS)
    s1 = avail * jnp.exp(1.8 * u0)
    s1_norm = s1 / jnp.sum(s1, axis=0, keepdims=True)
    l1 = s1_norm / jnp.sum(s1_norm, axis=1, keepdims=True)
    u1 = jnp.log(0.35 + 0.65 * l1 + EPS)

    scaled_sal = params["w_salience"] * salience
    prior = softmax_prior(scaled_sal)
    qud_prior = softmax_prior(scaled_sal + qud_log_mod)

    heard = L2(params["alpha"], avail, u1, qud_prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
