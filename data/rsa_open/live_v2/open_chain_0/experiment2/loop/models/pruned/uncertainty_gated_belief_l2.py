"""Depth-2 RSA listener and speaker with an uncertainty-gated belief decision rule and boundary-extremity prior."""

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
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., gate: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(
        r in OBJ,
        wpp=(1.0 - vec(gate, u)) * Pr[speaker.r == r] + vec(gate, u) * vec(prior, r),
    )
    return Pr[listener.r == r]


def compute_lex_prior_and_gate(ctx, w_salience):
    real_mask = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_mask
    diff = real_lex[:, :, None] - real_lex[:, None, :]
    pair_dist = jnp.sum(diff * diff, axis=0)
    is_singleton = jnp.where(jnp.sum(pair_dist <= EPS, axis=1) == 1, 1.0, 0.0)
    distinct = jnp.mean(pair_dist, axis=1)

    n_real = jnp.maximum(jnp.sum(1.0 - ctx.is_sink), 1.0)
    fc_norm = ctx.feature_count / n_real
    fc_dev = fc_norm - jnp.mean(fc_norm)
    fc_min = jnp.min(fc_norm)
    fc_max = jnp.max(fc_norm)
    extremity = (fc_dev * fc_dev - (fc_norm - fc_min) * (fc_max - fc_norm)) / (
        fc_norm + 1.0 / n_real
    )

    prior = softmax_prior(w_salience * (is_singleton + distinct + extremity))

    n_utt = ctx.lex.shape[0]
    n_obj = ctx.lex.shape[1]
    sem_noise = 1.0 / (n_utt * n_obj)
    fc_safe = jnp.maximum(ctx.feature_count, 1.0)[None, :]
    graded_true = ctx.lex / fc_safe
    noisy_real = graded_true + sem_noise * (1.0 - ctx.lex)
    soft_lex = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, noisy_real)

    ext_size = jnp.sum(real_lex, axis=1)
    false_size = n_obj - ext_size
    overload = jnp.maximum(ext_size - false_size, 0.0) * jnp.minimum(false_size, 1.0) / n_obj
    gate = overload * overload
    return soft_lex, prior, gate


def listener(params, ctx):
    soft_lex, prior, gate = compute_lex_prior_and_gate(ctx, params["w_salience"])
    heard = L2(params["alpha"], soft_lex, prior, gate)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    soft_lex, prior, gate = compute_lex_prior_and_gate(ctx, params["w_salience"])
    table = L2(params["alpha"], soft_lex, prior, gate)
    real = (1.0 - ctx.is_sink)[:, None]
    weights = (table + EPS) * real
    return weights / jnp.sum(weights, axis=0, keepdims=True)
