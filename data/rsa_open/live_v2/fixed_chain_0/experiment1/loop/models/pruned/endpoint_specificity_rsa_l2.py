import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_endpoint": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., spec_bonus: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS}) + at(spec_bonus, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., spec_bonus: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, lex, prior, spec_bonus) + {EPS})
                + at(spec_bonus, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def S2[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., spec_bonus: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r)
        * exp(
            alpha * log(L1[u, r](alpha, lex, prior, spec_bonus) + {EPS})
            + at(spec_bonus, u, r)
        ),
    )
    return Pr[speaker.u == u]


def _components(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diffs = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diffs, axis=0)
    duplicate_count = jnp.sum(pair_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)
    distinct = jnp.sum(pair_dist, axis=1)

    fc = ctx.feature_count
    min_fc = jnp.min(fc)
    max_fc = jnp.max(fc)
    is_endpoint = jnp.where((fc == min_fc) | (fc == max_fc), 1.0, 0.0)

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * fc
        + params["w_endpoint"] * is_endpoint
    )

    ext = jnp.sum(real_lex, axis=1, keepdims=True)
    is_shared_real = jnp.where((ext > 1.0) & (ctx.is_sink[:, None] == 0.0), 1.0, 0.0)
    spec_bonus = -params["w_endpoint"] * is_shared_real * (fc[None, :] - 1.0)
    return prior, spec_bonus


def listener(params, ctx):
    prior, spec_bonus = _components(params, ctx)
    heard = L2(params["alpha"], ctx.lex, prior, spec_bonus)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)


def speaker(params, ctx):
    prior, spec_bonus = _components(params, ctx)
    raw = S2(params["alpha"], ctx.lex, prior, spec_bonus)
    real = (1.0 - ctx.is_sink)[:, None]
    weights = raw * real
    total = weights.sum(axis=0, keepdims=True)
    indifferent = jnp.broadcast_to(real / real.sum(), weights.shape)
    return jnp.where(total > 0, weights / jnp.where(total > 0, total, 1.0), indifferent)
