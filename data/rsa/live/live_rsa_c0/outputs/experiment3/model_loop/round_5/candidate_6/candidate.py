"""Subsumption exemplar listener: pragmatic listener under capacity-limited executive
attention reasoning about speakers sensitive to contrastive feature subsumption hierarchies,
with exemplar-based contextual category typicality under the Generalized Context Model.

Refining contrastive_subsumption_listener by replacing contextual category typicality from
additive feature overlap with exemplar-based psychological similarity under the Generalized
Context Model from exemplar_typicality_listener: listeners evaluate contextual typicality as
an object's average exponential-decay similarity to all other scene exemplars across the visual
display rather than an unnormalized sum of shared feature fractions. This directly rectifies the
observed under-selection of duplicate-matching competitors in three-object displays
(three_object_target_vs_duplicate_competitor_rate), the under-selection of full duplicates on
uninformative mumble trials (mumble_full_duplicate_choice_rate), and the excessive singleton bias
(mumble_singleton_choice_rate), while fully preserving contrastive feature subsumption avoidance,
capacity-limited executive attention, and contrastive outlier pop-out.

Differences from contrastive_subsumption_listener:
- Recursion depth remains depth 2 (choice_probs calls L2, which inverts a speaker simulating depth-1 listener L1).
- Added parameters: None.
- Removed parameters: None (retains alpha, c_subsume, logit_p_base, w_load, w_popout, w_candidate, logit_p_outlier, w_inhibit_leak, w_typical, w_singleton, w_distinct, w_features, w_valence, w_color, w_confusion, and lapse with identical priors).
- In choice_probs: replaced compute_typicality(ctx) with an exemplar-based psychological similarity formulation computing an object's contextual typicality as its mean pairwise exponential-decay psychological similarity to all other scene exemplars (the sum of exp(-dist) to all other scene objects divided by N_obj - 1) instead of the unnormalized sum of shared feature fractions.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

MODES = jnp.arange(2)

PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "c_subsume": dist.Normal(0.0, 1.0),
    "logit_p_base": dist.Normal(2.0, 1.0),
    "w_load": dist.Normal(0.5, 0.5),
    "w_popout": dist.Normal(1.0, 0.5),
    "w_candidate": dist.Normal(0.5, 0.5),
    "logit_p_outlier": dist.Normal(-2.5, 1.0),
    "w_inhibit_leak": dist.Normal(0.0, 1.0),
    "w_typical": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](
    alpha, w_confusion, lex: ..., prior: ..., confusion: ..., subsumed: ...
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * (L0[u, r](lex) - at(subsumed, u, r))
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
    subsumed: ...,
    capture_prior: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * (
                    L1[u, r](alpha, w_confusion, lex, prior, confusion, subsumed)
                    - at(subsumed, u, r)
                )
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


def compute_typicality(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    diffs = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    dist = jnp.sum(diffs, axis=0)
    n_obj = ctx.lex.shape[1]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)
    return jnp.sum(sim, axis=1) / jnp.maximum(1.0, n_obj - 1.0)


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


def compute_subsumption(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    fc = jnp.sum(real_lex, axis=0)
    diff_i_not_j = jnp.sum(
        real_lex[:, :, None] * (1.0 - real_lex[:, None, :]), axis=0
    )
    strictly_subsumes = (diff_i_not_j == 0) & (fc[None, :] > fc[:, None])
    strictly_subsumes = strictly_subsumes.astype(jnp.float32)

    matches = (real_lex > 0).astype(jnp.float32)
    masked_fc = jnp.where(matches > 0, fc[None, :], 999.0)
    min_fc = jnp.min(masked_fc, axis=1, keepdims=True)

    is_intermediate = (matches > 0) & (fc[None, :] > min_fc)
    subsumed_by_matching = strictly_subsumes[None, :, :] * matches[:, None, :]
    has_subsumer = jnp.max(subsumed_by_matching, axis=2)

    subsumed_matrix = (
        is_intermediate.astype(jnp.float32) * has_subsumer * real_words
    )
    return subsumed_matrix


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

    effective_w_features = (
        params["w_features"] + ctx.valence * params["w_valence"]
    )

    typicality = compute_typicality(ctx)

    salience_logits = (
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + effective_w_features * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
        + params["w_typical"] * typicality
    )
    prior = softmax_prior(salience_logits)

    confusion = compute_confusion(ctx)
    subsumed = params["c_subsume"] * compute_subsumption(ctx)

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
        subsumed,
        capture_prior,
    )[ctx.utterance]
    heard_with_popout = (
        (1.0 - p_outlier) * heard + p_outlier * solitary_foil_vec
    )

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard_with_popout), params["lapse"]
    )
