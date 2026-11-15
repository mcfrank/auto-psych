"""Depth-2 RSA listener and speaker reasoning under a QUD of aspectual feature alignment."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_salience": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, l0_lex: ..., s1_lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(s1_lex, u, r) * exp(alpha * log(L0[u, r](l0_lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, l0_lex: ..., s1_lex: ..., s2_lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(s2_lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, l0_lex, s1_lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_qud_terms(ctx, w_salience):
    real_mask = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_mask
    diff = real_lex[:, :, None] - real_lex[:, None, :]
    pair_dist = jnp.sum(diff * diff, axis=0)
    is_singleton = jnp.where(jnp.sum(pair_dist <= EPS, axis=1) == 1, 1.0, 0.0)
    distinct = jnp.mean(pair_dist, axis=1)

    n_real = jnp.maximum(jnp.sum(1.0 - ctx.is_sink), 1.0)
    fc_norm = ctx.feature_count / n_real
    fc_dev = fc_norm - jnp.mean(fc_norm)
    extremity = fc_dev * fc_dev / (fc_norm + 1.0 / n_real)

    prior = softmax_prior(w_salience * (is_singleton + distinct + extremity))

    n_utt = ctx.lex.shape[0]
    n_obj = ctx.lex.shape[1]
    ext_size = jnp.sum(real_lex, axis=1, keepdims=True)
    fc_safe = jnp.maximum(ctx.feature_count, 1.0)[None, :]
    graded_true = ctx.lex / fc_safe
    cooccur = real_lex @ graded_true.T @ graded_true
    ambig = jnp.maximum(ext_size - 1.0, 0.0)

    noisy_real = (
        graded_true
        + (ambig / (n_obj * n_obj)) * cooccur * ctx.lex
        + ((ext_size * ext_size) / (n_utt * n_obj * n_obj)) * (1.0 - ctx.lex)
    )
    soft_lex = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, noisy_real)
    return ctx.lex, soft_lex, ctx.lex, prior


def listener(params, ctx):
    l0_lex, s1_lex, s2_lex, prior = compute_qud_terms(ctx, params["w_salience"])
    heard = L2(params["alpha"], l0_lex, s1_lex, s2_lex, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    l0_lex, s1_lex, s2_lex, prior = compute_qud_terms(ctx, params["w_salience"])
    table = L2(params["alpha"], l0_lex, s1_lex, s2_lex, prior)
    real = (1.0 - ctx.is_sink)[:, None]
    weights = (table + EPS) * real
    return weights / jnp.sum(weights, axis=0, keepdims=True)
