"""Pragmatic listener at depth 2 with diagnostic lexical competition, absence-oddball topic salience, and crowded-utterance L0 and prior retention."""

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
            wpp=at(avail, u, r) * exp(alpha * log(L1[u, r](alpha, avail, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask
    n_obj = ctx.lex.shape[1]

    # Summed exponential dissimilarity gated by positive feature presence, with competitor-normalized
    # baseline and high-density absence-oddball pop-out for featureless objects.
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    mean_dist = jnp.sum(pair_dist, axis=1) / jnp.maximum(n_obj - 1.0, 1.0)
    plain_gate = 1.5 / jnp.maximum(n_obj - 1.0, 1.0) + 0.55 * jnp.maximum(mean_dist - 2.0, 0.0)
    pos_gate = jnp.where(ctx.feature_count == 0, plain_gate, 1.0 - 0.25 * jnp.exp(-ctx.feature_count))
    contrast = 0.5 * pos_gate * jnp.sum(1.0 - jnp.exp(-pair_dist), axis=1)

    salience = contrast + (0.22 + 0.50 * ctx.valence) * ctx.feature_count
    prior = softmax_prior(params["w_salience"] * salience)

    # Diagnostic lexical competition during speaker retrieval (only narrow features covering <=2 objects compete)
    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    diag_w = jnp.where(feat_ext <= 2.0, 1.0, 0.0) * real_mask
    diag_fc = jnp.sum(real_lex * diag_w, axis=0, keepdims=True)
    fc_excess = jnp.maximum(diag_fc - 1.0, 0.0)
    true_avail = 1.0 / (1.0 + 0.5 * fc_excess)
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * (ctx.lex * true_avail)

    prag = L2(params["alpha"], avail, prior)[ctx.utterance]
    lit0 = L0(avail)[ctx.utterance]

    # When the heard word is over-extended across >2 objects, the listener partially falls back
    # to depth-0 lexical-competition grounding L0(avail) alongside retaining a small fraction of the topic prior.
    heard_ext = jnp.sum(real_lex[ctx.utterance])
    overext = jnp.maximum(heard_ext - 2.0, 0.0)
    ret_l0 = 0.32 * overext
    ret_prior = 0.0065 * overext
    heard = (1.0 - ret_l0 - ret_prior) * prag + ret_l0 * lit0 + ret_prior * prior

    return jnp.where(ctx.is_prior > 0, prior, heard)
