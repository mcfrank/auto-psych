"""Contrastive outlier listener: pragmatic listener balancing informativeness against contrastive pop-out.

Listeners interpret referring expressions by balancing top-down pragmatic informativeness
against bottom-up contrastive outlier pop-out. When an utterance is true of all objects in a
scene except one, the mentioned property highlights a shared background that isolates the
solitary non-matching object as an oddity along that feature dimension; listeners occasionally
select this contrastive outlier.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "logit_p_outlier": dist.Normal(-2.5, 1.0),
    "w_singleton": dist.Normal(1.0, 0.5),
    "w_distinct": dist.Normal(0.15, 0.2),
    "w_features": dist.Normal(0.2, 0.2),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_solitary_foil(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    matching_count = jnp.sum(real_lex, axis=1)
    n_obj = ctx.lex.shape[1]
    eligible_utt = (
        (matching_count == (n_obj - 1.0))
        & (n_obj >= 3.0)
        & (ctx.is_sink == 0.0)
    )
    is_foil = (ctx.lex == 0.0) * eligible_utt[:, None] * real_words
    return is_foil.astype(jnp.float32)


def compute_salience(ctx, params):
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

    salience_logits = (
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
    )
    return softmax_prior(salience_logits)


def choice_probs(params, ctx):
    prior = compute_salience(ctx, params)
    is_foil = compute_solitary_foil(ctx)
    solitary_foil_vec = is_foil[ctx.utterance]
    has_foil = jnp.max(solitary_foil_vec)
    p_outlier = jax.nn.sigmoid(params["logit_p_outlier"]) * has_foil

    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    heard_with_popout = (1.0 - p_outlier) * heard + p_outlier * solitary_foil_vec
    probs = jnp.where(ctx.is_prior > 0, prior, heard_with_popout)
    return with_lapse(probs, params["lapse"])
