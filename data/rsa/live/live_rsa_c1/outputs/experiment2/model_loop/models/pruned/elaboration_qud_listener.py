"""Pragmatic listener reasoning about speaker question under discussion: referent identity vs visual elaboration category."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "p_qud": dist.Beta(1.0, 5.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_extremity": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}

GOALS = (0, 1)


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, p_qud, lex: ..., prior: ..., util_elab: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            g in GOALS,
            wpp=(1.0 - p_qud) if g == 0 else p_qud,
        ),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * (
                    (1.0 - g) * log(L0[u, r](lex) + {EPS})
                    + g * at(util_elab, u, r)
                )
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    duplicate_count = jnp.sum(pair_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    base_logits = (
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
    )

    min_count = jnp.min(ctx.feature_count)
    max_count = jnp.max(ctx.feature_count)
    midpoint = 0.5 * (min_count + max_count)
    extremity = jnp.abs(ctx.feature_count - midpoint)

    speaker_prior = softmax_prior(base_logits)
    mumble_prior = softmax_prior(base_logits + params["w_extremity"] * extremity)

    l0 = ctx.lex / (jnp.sum(ctx.lex, axis=1, keepdims=True) + EPS)
    eq_cat = (ctx.feature_count[:, None] == ctx.feature_count[None, :]).astype(float)
    p_cat = jnp.einsum("ur, cr -> uc", l0, eq_cat)
    util_elab = jnp.log(p_cat + EPS)

    heard = L1(
        params["alpha"], params["p_qud"], ctx.lex, speaker_prior, util_elab
    )[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, mumble_prior, heard)
    return with_lapse(choice, params["lapse"])
