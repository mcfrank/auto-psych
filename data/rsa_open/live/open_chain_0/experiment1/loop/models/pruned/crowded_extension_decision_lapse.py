"""Pragmatic listener with diagnostic-feature lexical competition and a crowded-extension structured decision lapse."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_contrast": dist.LogNormal(0.0, 0.5),
    "w_features": dist.Normal(0.0, 1.0),
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

    # Pairwise feature distance across competitors
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = (
        jnp.sum(diff, axis=0)
        + jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
        + jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    )

    # Featureless objects are attenuated in sparse scenes but pop out as blank oddballs
    # when surrounded by complex competitors (mean distance > 2 features).
    is_plain = jnp.where(ctx.feature_count == 0, 1.0, 0.0)
    mean_dist = jnp.sum(pair_dist, axis=1) / jnp.maximum(n_obj - 1.0, 1.0)
    pos_gate = (
        1.0
        - 0.30 * is_plain
        + 0.65 * is_plain * jnp.maximum(mean_dist - 2.0, 0.0)
    )
    contrast = pos_gate * jnp.sum(1.0 - jnp.exp(-pair_dist), axis=1) / jnp.maximum(n_obj - 1.0, 1.0)

    effective_w_feat = params["w_features"] * (1.0 + 1.0 * ctx.valence)
    prior = softmax_prior(
        params["w_contrast"] * contrast
        + effective_w_feat * ctx.feature_count
    )

    # Diagnostic feature competition during speaker lexical access:
    # narrower features (ext <= 2) compete strongly, whereas broad features (ext > 2) compete weakly.
    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    feat_diag_wt = jnp.where(feat_ext <= 2.0, 1.0, 0.25) * real_mask
    diag_load = jnp.sum(real_lex * feat_diag_wt, axis=0, keepdims=True)
    true_avail = 1.0 / (1.0 + 0.50 * jnp.maximum(diag_load - 1.0, 0.0))
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * (ctx.lex * true_avail)

    prag = L2(params["alpha"], avail, prior)[ctx.utterance]

    # Decision rule: when the heard word is crowded across >2 objects, structured lapse shifts
    # belief toward equal probability matching over literal candidates and a small prior spillover.
    heard_ext = jnp.sum(real_lex[ctx.utterance])
    overext = jnp.maximum(heard_ext - 2.0, 0.0)
    lit_match = real_lex[ctx.utterance] / jnp.maximum(heard_ext, 1.0)
    lapse_lit = 0.28 * overext
    lapse_prior = 0.009 * overext
    heard = (1.0 - lapse_lit - lapse_prior) * prag + lapse_lit * lit_match + lapse_prior * prior

    return jnp.where(ctx.is_prior > 0, prior, heard)
