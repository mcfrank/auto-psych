import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ..., deficit_sq: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - beta * at(deficit_sq, u, r)
            ),
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


def compute_deficit_sq(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_obj = ctx.lex.shape[1]
    ext = jnp.sum(real_lex, axis=1)
    contrast_u = jnp.where(
        ctx.is_sink > 0,
        0.0,
        (n_obj - ext) / jnp.maximum(n_obj - 1.0, 1.0),
    )
    obj_word_contrast = real_lex * contrast_u[:, None]
    max_contrast_r = jnp.max(obj_word_contrast, axis=0)
    diff = max_contrast_r[None, :] - contrast_u[:, None]
    deficit = jnp.maximum(0.0, diff) / (max_contrast_r[None, :] + 1e-4)
    return (deficit * real_lex) ** 2


def listener(params, ctx):
    is_singleton = compute_singleton(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    deficit_sq = compute_deficit_sq(ctx)
    heard = L1(
        params["alpha"], params["beta"], ctx.lex, deficit_sq, prior
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
