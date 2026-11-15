"""Depth-2 RSA theory with aspectual QUD retrieval in production and graded lexical uncertainty in comprehension."""

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
def L2[u: UTT, r: OBJ](alpha, s2_lex: ..., lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(s2_lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def S1[u: UTT, r: OBJ](alpha, true_lex: ..., spec_lex: ..., ext_share: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(spec_lex, u, r)
        * exp((1.0 + alpha) * (log(L0[u, r](true_lex) + {EPS}) - vec(ext_share, u))),
    )
    return Pr[speaker.u == u]


@memo
def S_prod[u: UTT, r: OBJ](
    alpha, true_lex: ..., spec_lex: ..., graded_lex: ..., fc_inv: ..., ext_share: ...
):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=S1[u, r](alpha, true_lex, spec_lex, ext_share)
        + vec(fc_inv, r) * (L0[u, r](true_lex) + L0[u, r](graded_lex)),
    )
    return Pr[speaker.u == u]


def compute_lex_and_prior(ctx, w_salience):
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
    ext_safe = jnp.maximum(ext_size, 1.0)
    ext_share = (ext_size[:, 0] / n_obj) * (1.0 - ctx.is_sink)
    denom_scale = jnp.maximum(n_utt, n_obj) * n_obj * n_obj
    sem_noise = (ext_size * ext_size) / denom_scale
    fc_safe = jnp.maximum(ctx.feature_count, 1.0)
    fc_inv = 1.0 / fc_safe
    graded_true = ctx.lex * fc_inv[None, :]

    disc_share = (n_obj + 1.0 - ext_safe) / (n_obj * ext_safe)
    spec_real = ctx.lex * disc_share * disc_share
    spec_lex = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, spec_real)

    noisy_real = graded_true + sem_noise * (1.0 - ctx.lex)
    s2_real = graded_true + (ext_size / n_obj) * sem_noise * (1.0 - ctx.lex)
    soft_lex = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, noisy_real)
    s2_lex = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, s2_real)
    return s2_lex, soft_lex, spec_lex, graded_true, fc_inv, ext_share, prior


def listener(params, ctx):
    s2_lex, soft_lex, _, _, _, _, prior = compute_lex_and_prior(ctx, params["w_salience"])
    heard = L2(params["alpha"], s2_lex, soft_lex, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    _, _, spec_lex, graded_true, fc_inv, ext_share, _ = compute_lex_and_prior(
        ctx, params["w_salience"]
    )
    table = S_prod(params["alpha"], ctx.lex, spec_lex, graded_true, fc_inv, ext_share)
    real = (1.0 - ctx.is_sink)[:, None]
    weights = (table + EPS) * real
    return weights / jnp.sum(weights, axis=0, keepdims=True)
