import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_contrast": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_contrast, lex: ..., contrast: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                + w_contrast * at(contrast, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_contrast(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_obj = ctx.lex.shape[1]
    sim = (real_lex.T @ real_lex) * (1.0 - jnp.eye(n_obj))
    not_lex = 1.0 - real_lex
    contrast = real_words * ((not_lex @ sim.T) * real_lex)
    return contrast


def compute_singleton(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None, None]
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :]) * real_words
    pair_dist = jnp.sum(diffs, axis=0)
    duplicate_count = jnp.sum(pair_dist == 0, axis=1)
    return jnp.where(duplicate_count == 1, 1.0, 0.0)


def compute_distinct(ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    return jnp.sum(diff, axis=(0, 2))


def listener(params, ctx):
    is_singleton = compute_singleton(ctx)
    distinct = compute_distinct(ctx)
    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
    )
    contrast = compute_contrast(ctx)
    heard = L1(
        params["alpha"],
        params["w_contrast"],
        ctx.lex,
        contrast,
        prior,
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
