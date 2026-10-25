"""Pragmatic listener reasoning under limited visual attention and contextual feature saliency.

Listeners interpret referring expressions under limited visual attention across
display features: in a visual scene, attention is captured by contextually distinctive
features that pop out against the display, so that common features shared broadly
across competitors are attentionally discounted. When evaluating literal semantics,
listeners weight an object's compatibility by the heard feature's share of that object's
contextual visual salience, expecting speakers to refer using an object's most distinctive
properties rather than ubiquitous descriptors. Communicative speakers choose expressions
anticipating this contrastive feature-attribution inference, while spontaneous visual
attention on uninformative displays defaults to contextually unique singletons.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
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


def compute_lex_norm(ctx):
    real_mask = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_mask
    feat_freq = jnp.sum(real_lex, axis=1)
    feat_rarity = jnp.where(feat_freq > 0, 1.0 / feat_freq, 0.0)
    weighted_feats = real_lex * feat_rarity[:, None]
    obj_contrast = jnp.sum(weighted_feats, axis=0)
    safe_contrast = jnp.where(obj_contrast > 0, obj_contrast, 1.0)
    norm_real = weighted_feats / safe_contrast[None, :]
    return jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, norm_real)


def listener(params, ctx):
    is_singleton = compute_singleton(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    lex_norm = compute_lex_norm(ctx)
    heard = L1(params["alpha"], ctx.lex, lex_norm, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
