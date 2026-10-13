"""Perceptual grouping listener: pragmatic listener under capacity-limited executive
attention reasoning about speakers with a Gestalt perceptual grouping prior over visual types.

Listeners interpret referring expressions under Gestalt perceptual grouping across identical
visual candidates: when duplicate objects appear in a visual scene, observers group identical
tokens into a single perceptual type rather than independent spatial individuals. Before hearing
an informative word and during referent resolution, listeners expect speakers to choose
communicative referents at the level of distinct visual types, allocating a prior over types
that is shared equally among identical tokens rather than compounding with spatial token
numerosity. This eliminates artificial singleton inflation on uninformative mumble trials,
properly calibrates choices between unique targets and duplicated competitors in both word
and mumble conditions, and applies the discourse prior in a principled place at the pragmatic
decision tier without recursive over-compounding.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

MODES = jnp.arange(2)

PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "w_group": dist.Normal(1.0, 0.5),
    "w_features": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 1.0),
    "logit_p_base": dist.Normal(2.0, 1.0),
    "w_load": dist.Normal(0.5, 0.5),
    "w_popout": dist.Normal(1.0, 0.5),
    "w_candidate": dist.Normal(0.5, 0.5),
    "logit_p_outlier": dist.Normal(-2.5, 1.0),
    "w_inhibit_leak": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) + {EPS})
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](
    alpha,
    w_confusion,
    lex: ...,
    confusion: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * L0[u, r](lex)
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](
    alpha,
    w_confusion,
    p_attend,
    lex: ...,
    prior: ...,
    confusion: ...,
    capture_prior: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * L1[u, r](alpha, w_confusion, lex, confusion)
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(
        mode in MODES, wpp=p_attend if mode == 1 else (1.0 - p_attend)
    )
    listener: chooses(
        r in OBJ,
        wpp=Pr[speaker.r == r] if mode == 1 else vec(capture_prior, r),
    )
    return Pr[listener.r == r]


def compute_confusion(ctx):
    feats = jnp.vstack(
        [ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]]
    )
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)

    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    return real_lex @ sim


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


def compute_prior(ctx, params):
    # Pairwise feature distance across all contextual dimensions to identify duplicate tokens
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1).astype(jnp.float32)

    # Contextual distinctiveness
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    effective_w_features = (
        params["w_features"] + ctx.valence * params["w_valence"]
    )

    # Perceptual grouping: duplicate tokens share type-level prior weight
    # Numerosity compression penalty: - w_group * log(duplicate_count)
    grouping_penalty = params["w_group"] * jnp.log(duplicate_count)

    salience_logits = (
        effective_w_features * ctx.feature_count
        + params["w_distinct"] * distinct
        - grouping_penalty
        + params["w_color"] * (1.0 - ctx.grayscale)
    )
    return softmax_prior(salience_logits)


def choice_probs(params, ctx):
    prior = compute_prior(ctx, params)
    confusion = compute_confusion(ctx)

    real_matches = ctx.lex[ctx.utterance] * (1.0 - ctx.is_sink[ctx.utterance])
    matching_count = jnp.sum(real_matches)
    is_popout = jnp.where(matching_count == 1.0, 1.0, 0.0)
    candidate_load = jnp.maximum(0.0, matching_count - 1.0)

    n_obj = ctx.lex.shape[1]
    logit_p_attend = (
        params["logit_p_base"]
        + params["w_popout"] * is_popout
        - params["w_load"] * (n_obj - 2.0)
        - params["w_candidate"] * candidate_load
    )
    p_attend = jax.nn.sigmoid(logit_p_attend)

    inhibit_leak = jax.nn.softplus(params["w_inhibit_leak"]) * candidate_load
    inhibit_factor = jnp.where(real_matches > 0, 1.0, inhibit_leak)
    capture_weights = prior * inhibit_factor + 1e-6
    capture_prior = capture_weights / jnp.sum(capture_weights)

    is_foil = compute_solitary_foil(ctx)
    solitary_foil_vec = is_foil[ctx.utterance]
    has_foil = jnp.max(solitary_foil_vec)
    p_outlier = jax.nn.sigmoid(params["logit_p_outlier"]) * has_foil

    heard = L2(
        params["alpha"],
        params["w_confusion"],
        p_attend,
        ctx.lex,
        prior,
        confusion,
        capture_prior,
    )[ctx.utterance]
    heard_with_popout = (1.0 - p_outlier) * heard + p_outlier * solitary_foil_vec

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard_with_popout), params["lapse"]
    )
