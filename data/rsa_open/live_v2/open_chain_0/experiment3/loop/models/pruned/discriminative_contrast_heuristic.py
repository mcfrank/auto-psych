"""Non-Bayesian discriminative contrast heuristic listener and speaker."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_salience": dist.Normal(0.0, 1.0),
}


@memo
def L_heur[u: UTT, r: OBJ](alpha, match_score: ..., contrast_score: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r)
        * at(match_score, u, r)
        * exp(alpha * at(contrast_score, u, r)),
    )
    return Pr[listener.r == r]


@memo
def S_heur[u: UTT, r: OBJ](alpha, spec_lex: ..., feat_diag: ..., ext_share: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(spec_lex, u, r)
        * exp((1.0 + alpha) * (log(at(feat_diag, u, r) + {EPS}) - vec(ext_share, u))),
    )
    return Pr[speaker.u == u]


@memo
def S_prod[u: UTT, r: OBJ](
    alpha, spec_lex: ..., feat_diag: ..., avail_lex: ..., ext_share: ...
):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=S_heur[u, r](alpha, spec_lex, feat_diag, ext_share) + at(avail_lex, u, r),
    )
    return Pr[speaker.u == u]


def compute_heuristic_terms(ctx, w_salience):
    real_mask = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_mask
    n_utt = ctx.lex.shape[0]
    n_obj = ctx.lex.shape[1]

    # Perceptual anchor salience: singleton pop-out + pairwise distinctiveness + complexity extremity
    diff = real_lex[:, :, None] - real_lex[:, None, :]
    pair_dist = jnp.sum(diff * diff, axis=0)
    is_singleton = jnp.where(jnp.sum(pair_dist <= EPS, axis=1) == 1, 1.0, 0.0)
    distinct = jnp.mean(pair_dist, axis=1)

    n_real = jnp.maximum(jnp.sum(1.0 - ctx.is_sink), 1.0)
    fc_norm = ctx.feature_count / n_real
    fc_dev = fc_norm - jnp.mean(fc_norm)
    extremity = fc_dev * fc_dev / (fc_norm + 1.0 / n_real)
    prior = softmax_prior(w_salience * (is_singleton + distinct + extremity))

    # First- and second-order feature diagnosticity
    ext_size = jnp.sum(real_lex, axis=1, keepdims=True)
    ext_safe = jnp.maximum(ext_size, 1.0)
    feat_diag = ctx.lex / ext_safe

    fc_safe = jnp.maximum(ctx.feature_count, 1.0)[None, :]
    graded_true = ctx.lex / fc_safe
    denom_scale = jnp.maximum(n_utt, n_obj) * n_obj * n_obj
    sem_noise = (ext_size * ext_size) / denom_scale

    noisy_real = graded_true + sem_noise * (1.0 - ctx.lex)
    soft_lex = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, noisy_real)
    s2_real = graded_true + (ext_size / n_obj) * sem_noise * (1.0 - ctx.lex)
    match_score = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, s2_real)

    # Non-Bayesian heuristic diagnostic contrast for comprehension
    soft_ext = jnp.maximum(jnp.sum(soft_lex, axis=1, keepdims=True), EPS)
    diag_0 = soft_lex / soft_ext
    obj_diag_0 = jnp.maximum(jnp.sum(diag_0 * soft_lex, axis=0, keepdims=True), EPS)
    rel_1 = (soft_lex * diag_0) / obj_diag_0
    eff_ext = jnp.maximum(jnp.sum(rel_1 * prior[None, :], axis=1, keepdims=True), EPS)
    diag_1 = (rel_1 * prior[None, :]) / eff_ext
    contrast_score = jnp.log(diag_1 + EPS)

    # Direct discriminative contrast terms for heuristic speaker production
    ext_share = (ext_size[:, 0] / n_obj) * (1.0 - ctx.is_sink)
    disc_share = (n_obj + 1.0 - ext_safe) / (n_obj * ext_safe)
    spec_real = ctx.lex * disc_share * disc_share
    spec_lex = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, spec_real)

    graded_ext = jnp.maximum(jnp.sum(graded_true, axis=1, keepdims=True), EPS)
    avail_lex = (graded_true / fc_safe) * (graded_true / graded_ext) / (1.0 + ext_share[:, None])

    return match_score, contrast_score, prior, spec_lex, feat_diag, avail_lex, ext_share


def listener(params, ctx):
    match_score, contrast_score, prior, _, _, _, _ = compute_heuristic_terms(
        ctx, params["w_salience"]
    )
    heard = L_heur(params["alpha"], match_score, contrast_score, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    _, _, _, spec_lex, feat_diag, avail_lex, ext_share = compute_heuristic_terms(
        ctx, params["w_salience"]
    )
    table = S_prod(params["alpha"], spec_lex, feat_diag, avail_lex, ext_share)
    real = (1.0 - ctx.is_sink)[:, None]
    weights = (table + EPS) * real
    return weights / jnp.sum(weights, axis=0, keepdims=True)
