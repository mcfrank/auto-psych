import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior

PARAMS = {
    "w_diag": dist.LogNormal(0.0, 0.5),
    "w_crowd": dist.LogNormal(0.0, 0.5),
}


@memo
def L_heur[u: UTT, r: OBJ](lex: ..., score: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * exp(at(score, u, r)))
    return Pr[listener.r == r]


def compute_heuristic_scores(ctx, w_diag, w_crowd):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist_mat = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist_mat.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist_mat) * (1.0 - eye)
    crowd = jnp.sum(sim, axis=1)
    copy_count = jnp.sum(jnp.where(dist_mat < EPS, 1.0, 0.0), axis=1)
    is_singleton = jnp.where(copy_count <= 1.0, 1.0, 0.0)

    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_real_utt = jnp.sum(1.0 - ctx.is_sink)

    type_ext = jnp.maximum(
        jnp.sum(real_lex / copy_count[None, :], axis=1, keepdims=True), 1.0
    )
    feat_diag = real_lex / type_ext
    n_utt = real_lex.shape[0]
    utt_eye = jnp.eye(n_utt)[:, :, None]
    other_diag = feat_diag[:, None, :] * (1.0 - utt_eye)
    best_alt_diag = jnp.max(other_diag, axis=0)

    fc = ctx.feature_count
    min_fc = jnp.min(fc)
    max_fc = jnp.max(fc)
    is_inter_scene = jnp.where((fc > min_fc) & (fc < max_fc), 1.0, 0.0)
    is_max_fc = jnp.where((fc == max_fc) & (max_fc > min_fc), 1.0, 0.0)
    is_unique_max = is_max_fc * is_singleton

    masked_min_fc = jnp.where(real_lex > 0, fc[None, :], max_fc + 1.0)
    min_true_fc = jnp.min(masked_min_fc, axis=1, keepdims=True)
    masked_max_fc = jnp.where(real_lex > 0, fc[None, :], 0.0)
    max_true_fc = jnp.max(masked_max_fc, axis=1, keepdims=True)

    is_min_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] == min_true_fc) & (max_true_fc > min_true_fc),
        1.0,
        0.0,
    )
    is_partial_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] > min_true_fc) & (fc[None, :] < n_real_utt),
        1.0,
        0.0,
    )

    diag_penalty = best_alt_diag - feat_diag
    word_crowd = real_lex @ sim
    framing = w_diag * (ctx.valence * fc + ctx.familiarization - ctx.grayscale)

    prior_logits = (
        -w_crowd * (crowd + is_inter_scene + (1.0 - is_singleton) - is_unique_max)
        + framing
    )
    prior_choice = softmax_prior(prior_logits)

    word_score = (
        framing[None, :]
        - w_diag * diag_penalty
        - w_crowd * (crowd[None, :] + word_crowd + is_partial_utt - is_min_utt)
    )
    score = jnp.where(ctx.is_sink[:, None] > 0, 0.0, word_score)
    return score, prior_choice


def listener(params, ctx):
    score, prior_choice = compute_heuristic_scores(
        ctx, params["w_diag"], params["w_crowd"]
    )
    heard = L_heur(ctx.lex, score)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior_choice, heard)
