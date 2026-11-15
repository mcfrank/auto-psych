"""Depth-2 RSA listener and speaker reasoning under a contrastive aspect Question Under Discussion (QUD)."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_salience": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](qud_lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(qud_lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, qud_lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(qud_lex, u, r) * exp(alpha * log(L0[u, r](qud_lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, qud_lex: ..., contrast_lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(qud_lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, qud_lex) + {EPS}))
            + at(contrast_lex, u, r),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_qud_components(ctx, w_salience):
    real_mask = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_mask
    n_real = jnp.maximum(jnp.sum(1.0 - ctx.is_sink), 1.0)
    n_utt = ctx.lex.shape[0]
    n_obj = ctx.lex.shape[1]

    diff = real_lex[:, :, None] - real_lex[:, None, :]
    pair_dist = jnp.sum(diff * diff, axis=0)
    clone_count = jnp.sum(pair_dist <= EPS, axis=1)
    uniqueness = 1.0 / jnp.maximum(clone_count, 1.0)

    fc = ctx.feature_count
    dice_dist = pair_dist / (fc[:, None] + fc[None, :] + 1.0)
    distinct = jnp.mean(dice_dist, axis=1)

    fc_dev = fc - jnp.mean(fc)
    extremity = fc_dev * fc_dev / (fc + 1.0)

    prior = softmax_prior(w_salience * (uniqueness + distinct + extremity))

    # Positive aspect completeness: moderated by normalized unmentioned feature load
    # and competitor feature overlap in the word's extension, with baseline semantic noise.
    eye = jnp.eye(n_obj)
    comp_shared = (real_lex.T @ real_lex) * (1.0 - eye)
    comp_ov = real_lex @ comp_shared
    pos_aspect = ctx.lex / (1.0 + (fc[None, :] + comp_ov) / n_real)
    base_noise = 1.0 / (n_utt * n_obj)
    qud_real = pos_aspect + base_noise * (1.0 - ctx.lex)
    qud_lex = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, qud_real)

    # Contrastive QUD partition weight at S2: proportional to positive extension share (n_pos / n_obj)
    # and complement cell specificity (1 / n_neg), scaled by baseline matrix size (n_utt * n_obj).
    n_pos = jnp.sum(real_lex, axis=1, keepdims=True)
    n_neg = jnp.maximum(n_obj - n_pos, 1.0)
    contrast_weight = (n_pos / n_obj) / (n_utt * n_obj * n_neg)
    contrast_lex = (1.0 - ctx.lex) * real_mask * contrast_weight
    return qud_lex, contrast_lex, prior


def listener(params, ctx):
    qud_lex, contrast_lex, prior = compute_qud_components(
        ctx, params["w_salience"]
    )
    heard = L2(params["alpha"], qud_lex, contrast_lex, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
