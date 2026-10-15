import jax.numpy as jnp
import jax
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_empty": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def S1[u: UTT, r: OBJ](alpha, lex: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L1[u: UTT, r: OBJ](lex: ..., prior: ..., s1: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * vec(prior, r) * at(s1, u, r),
    )
    return Pr[listener.r == r]


@memo
def S2[u: UTT, r: OBJ](alpha, lex: ..., l1: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r) * exp(alpha * log(at(l1, u, r) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L2[u: UTT, r: OBJ](lex: ..., prior: ..., s2: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * vec(prior, r) * at(s2, u, r),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    is_empty = jnp.where(ctx.feature_count == 0, 1.0, 0.0)

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_empty"] * is_empty
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    s1 = S1(params["alpha"], ctx.lex)
    l1 = L1(ctx.lex, prior, s1)
    s2 = S2(params["alpha"], ctx.lex, l1)
    heard = L2(ctx.lex, prior, s2)[ctx.utterance]

    # Softmax over referential beliefs with response sensitivity gamma (Luce choice rule)
    is_match = ctx.lex[ctx.utterance] > 0
    log_heard = jnp.where(is_match, jnp.log(heard + EPS), -1e9)
    choice = jax.nn.softmax(params["gamma"] * log_heard)

    p_choice = (1.0 - params["lapse"]) * choice + params["lapse"] * prior
    return jnp.where(ctx.is_prior > 0, prior, p_choice)
