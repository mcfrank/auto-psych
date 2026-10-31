"""Non-Bayesian heuristic listener based on exclusive cue ownership and perceptual isolation."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "w_rival": dist.LogNormal(0.0, 0.5),
    "w_isolation": dist.LogNormal(0.0, 0.5),
    "w_ownership": dist.LogNormal(0.0, 0.5),
}


@memo
def L_heur[u: UTT, r: OBJ](w_rival, match: ..., rival: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(match, u, r) * vec(prior, r) * exp(-w_rival * at(rival, u, r)),
    )
    return Pr[listener.r == r]


def listener(params, ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask
    n_obj = ctx.lex.shape[1]

    # Each present feature divides its cue weight equally among the objects possessing it.
    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    cue_weight = real_mask / jnp.maximum(feat_ext, 1.0)
    obj_Cue = real_lex * cue_weight
    total_ownership = jnp.sum(obj_Cue, axis=0)

    # Unmentioned rival cue ownership when word u is heard.
    rival = total_ownership[None, :] - obj_Cue

    # Perceptual isolation: mean logarithmic Hamming distance across competitors,
    # gated by positive feature presence so featureless objects do not over-pop-out in sparse scenes.
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = (
        jnp.sum(diff, axis=0)
        + jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
        + jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    )
    pos_gate = 1.0 - 0.30 * jnp.exp(-ctx.feature_count)
    isolation = pos_gate * jnp.sum(jnp.log1p(pair_dist), axis=1) / jnp.maximum(n_obj - 1.0, 1.0)

    # Normalized cue ownership relative to scene feature count so dimensionality does not inflate feature bias.
    n_real_feat = jnp.maximum(jnp.sum(1.0 - ctx.is_sink), 1.0)
    norm_ownership = total_ownership / n_real_feat

    salience = (
        params["w_isolation"] * isolation
        + params["w_ownership"] * (1.0 + 1.5 * ctx.valence) * norm_ownership
    )
    prior = softmax_prior(salience)

    # Semantic match with crowded-extension leakage when a word covers >2 objects.
    overext = jnp.maximum(feat_ext - 2.0, 0.0)
    false_leak = (0.005 + 0.045 * overext) / n_obj
    real_match = ctx.lex + (1.0 - ctx.lex) * false_leak
    match = ctx.is_sink[:, None] * ctx.lex + real_mask * real_match

    heard = L_heur(params["w_rival"], match, rival, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
