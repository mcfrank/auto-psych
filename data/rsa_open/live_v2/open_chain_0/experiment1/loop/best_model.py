"""Depth-2 RSA listener with extension-scaled noisy graded lexical semantics and extremity-uniqueness salience."""

import jax
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


def compute_soft_lex_and_prior(ctx, w_salience):
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

    # Graded truth value: each true feature shares attentional capacity (1 / feature_count),
    # and false features receive semantic confusion scaled by the squared extension proportion (ext_size / n_obj)^2 / n_utt.
    n_utt = ctx.lex.shape[0]
    n_obj = ctx.lex.shape[1]
    ext_size = jnp.sum(real_lex, axis=1, keepdims=True)
    sem_noise = (ext_size * ext_size) / (n_utt * n_obj * n_obj)
    fc_safe = jnp.maximum(ctx.feature_count, 1.0)[None, :]
    graded_true = ctx.lex / fc_safe
    noisy_real = graded_true + sem_noise * (1.0 - ctx.lex)
    soft_lex = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, noisy_real)
    return soft_lex, prior


def listener(params, ctx):
    soft_lex, prior = compute_soft_lex_and_prior(ctx, params["w_salience"])
    heard = L2(params["alpha"], soft_lex, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    soft_lex, prior = compute_soft_lex_and_prior(ctx, params["w_salience"])
    table = L2(params["alpha"], soft_lex, prior)
    real = (1.0 - ctx.is_sink)[:, None]
    weights = (table + EPS) * real
    return weights / jnp.sum(weights, axis=0, keepdims=True)
