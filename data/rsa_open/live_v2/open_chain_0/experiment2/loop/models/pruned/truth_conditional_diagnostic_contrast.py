"""Non-Bayesian diagnostic contrast heuristic listener and speaker."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_salience": dist.Normal(0.0, 1.0),
}


@memo
def L_heur[u: UTT, r: OBJ](alpha, w_salience, match_score: ..., unm_diag: ..., salience: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(match_score, u, r)
        * exp(-alpha * at(unm_diag, u, r) + w_salience * vec(salience, r)),
    )
    return Pr[listener.r == r]


def compute_heuristic_terms(ctx):
    real_mask = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_mask

    # Per-feature diagnosticity: 1 / number of objects possessing the feature
    ext_size = jnp.maximum(jnp.sum(real_lex, axis=1, keepdims=True), 1.0)
    feat_diag = real_lex / ext_size

    # Second-order diagnosticity: how strongly each object relies on each feature
    obj_diag = jnp.maximum(jnp.sum(feat_diag, axis=0, keepdims=True), EPS)
    rel_diag = feat_diag / obj_diag
    eff_ext = jnp.maximum(jnp.sum(rel_diag, axis=1, keepdims=True), EPS)
    eff_diag = real_lex / eff_ext

    # Unmentioned feature diagnosticity (first-order + second-order), centered across valid referents
    comb_diag = feat_diag + eff_diag
    total_diag = jnp.sum(comb_diag, axis=0, keepdims=True)
    raw_unm = total_diag - comb_diag
    mean_unm = jnp.sum(raw_unm * ctx.lex, axis=1, keepdims=True) / jnp.maximum(
        jnp.sum(ctx.lex, axis=1, keepdims=True), 1.0
    )
    unm_diag = raw_unm - mean_unm

    # Truth-conditional match to the heard word: diluted across object's features and feature competition
    fc_safe = jnp.maximum(ctx.feature_count, 1.0)[None, :]
    true_match = ctx.lex / (fc_safe * eff_ext)
    match_score = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, true_match)

    # Perceptual anchor salience: singleton pop-out + pairwise distinctiveness + complexity extremity
    diff = real_lex[:, :, None] - real_lex[:, None, :]
    pair_dist = jnp.sum(diff * diff, axis=0)
    is_singleton = jnp.where(jnp.sum(pair_dist <= EPS, axis=1) == 1, 1.0, 0.0)
    distinct = jnp.mean(pair_dist, axis=1)

    n_real = jnp.maximum(jnp.sum(1.0 - ctx.is_sink), 1.0)
    fc_norm = ctx.feature_count / n_real
    fc_dev = fc_norm - jnp.mean(fc_norm)
    load = jnp.where(ctx.feature_count > 0, fc_norm, 1.0)
    extremity = fc_dev * fc_dev / (load + 1.0 / n_real)

    raw_sal = is_singleton + distinct + extremity
    salience = raw_sal - jnp.mean(raw_sal)
    return match_score, unm_diag, salience


def listener(params, ctx):
    match_score, unm_diag, salience = compute_heuristic_terms(ctx)
    prior = softmax_prior(params["w_salience"] * salience)
    heard = L_heur(
        params["alpha"], params["w_salience"], match_score, unm_diag, salience
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    match_score, unm_diag, salience = compute_heuristic_terms(ctx)
    table = L_heur(
        params["alpha"], params["w_salience"], match_score, unm_diag, salience
    )
    real = (1.0 - ctx.is_sink)[:, None]
    weights = (table + EPS) * real
    return weights / jnp.sum(weights, axis=0, keepdims=True)
