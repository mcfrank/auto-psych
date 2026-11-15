import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_extreme": dist.Normal(0.0, 1.0),
    "w_word_tier": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., tier: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r] * at(tier, u, r))
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., tier: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, prior, tier) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r] * at(tier, u, r))
    return Pr[listener.r == r]


def listener(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diffs = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diffs, axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)

    distinct = jnp.sum(diffs, axis=(0, 2))

    fc = ctx.feature_count
    is_min_fc = jnp.where(fc == jnp.min(fc), 1.0, 0.0)
    is_max_fc = jnp.where(fc == jnp.max(fc), 1.0, 0.0)
    is_extreme = jnp.where((is_min_fc + is_max_fc) > 0, 1.0, 0.0)

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_extreme"]
        * (is_extreme + is_max_fc + ctx.valence * fc + (1.0 - ctx.grayscale))
    )

    max_fc = jnp.max(fc) + 1.0
    masked_min_fc = jnp.where(ctx.lex > 0, fc[None, :], max_fc)
    min_true_fc = jnp.min(masked_min_fc, axis=1, keepdims=True)
    masked_max_fc = jnp.where(ctx.lex > 0, fc[None, :], -1.0)
    max_true_fc = jnp.max(masked_max_fc, axis=1, keepdims=True)

    is_min_true = jnp.where((ctx.lex > 0) & (fc[None, :] == min_true_fc), 1.0, 0.0)
    is_mid_true = jnp.where(
        (ctx.lex > 0) & (fc[None, :] > min_true_fc) & (fc[None, :] < max_true_fc),
        1.0,
        0.0,
    )
    tier_score = is_min_true - is_mid_true
    tier = jnp.where(
        ctx.is_sink[:, None] > 0,
        1.0,
        jnp.exp(params["w_word_tier"] * tier_score),
    )

    heard = L2(params["alpha"], ctx.lex, prior, tier)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
