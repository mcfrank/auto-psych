"""Chain 0's final, ``boundary_avoidable_ambiguity_listener``, with every mechanism switchable.

A memo model file (`src.rsa.model_file`), read as text and never imported: a variant model file
is this source plus a footer that calls ``build(MECHANISMS_ON)`` (`src.rsa.ablation.variants`).
With every mechanism on it is the final (data/rsa/live/finals/boundary_avoidable_ambiguity_listener.py),
expression for expression (`tests/test_rsa_c0_ablation.py`). The mechanisms, their parameters and
what removing each leaves are listed in `src.rsa.ablation.spec_c0`.

The final: a depth-2 listener (L2) reasoning about a speaker who reasons about L1, with a
salience prior over objects, an interior-suppression and an avoidable-ambiguity cost, a
confusion penalty; an attention gate whose lapse is captured by the prior over matching objects
(with an inhibition leak onto non-matching ones); a solitary-foil outlier; and a lapse.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

MODES = jnp.arange(2)

CORE_PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}
MECHANISM_PARAMS = {
    "depth2": {},
    "interior": {"w_interior": dist.Normal(0.0, 1.0)},
    "avoidable": {"c_avoid": dist.Normal(0.0, 1.0)},
    "confusion": {"w_confusion": dist.Normal(0.0, 1.0)},
    "singleton": {"w_singleton": dist.Normal(0.0, 1.0)},
    "distinct": {"w_distinct": dist.Normal(0.0, 1.0)},
    "features": {"w_features": dist.Normal(0.0, 1.0)},
    "valence": {"w_valence": dist.Normal(0.0, 1.0)},
    "color": {"w_color": dist.Normal(0.0, 1.0)},
    "typicality": {"w_typical": dist.Normal(0.0, 1.0)},
    "attention": {"logit_p_base": dist.Normal(2.0, 1.0)},
    "popout": {"w_popout": dist.Normal(1.0, 0.5)},
    "load": {"w_load": dist.Normal(0.5, 0.5)},
    "candidate": {"w_candidate": dist.Normal(0.5, 0.5)},
    "inhibit_leak": {"w_inhibit_leak": dist.Normal(0.0, 1.0)},
    "outlier": {"logit_p_outlier": dist.Normal(-2.5, 1.0)},
}
# The final's parameter order, so the full variant's PARAMS are the final's, in order.
FINAL_ORDER = ("alpha", "w_interior", "c_avoid", "logit_p_base", "w_load", "w_popout", "w_candidate",
               "logit_p_outlier", "w_inhibit_leak", "w_typical", "w_singleton", "w_distinct", "w_features",
               "w_valence", "w_color", "w_confusion", "lapse")


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) + {EPS})
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](
    alpha,
    w_confusion,
    w_interior,
    lex: ...,
    prior: ...,
    confusion: ...,
    interior: ...,
    avoidable: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * (
                    L0[u, r](lex)
                    - w_interior * at(interior, u, r)
                    - at(avoidable, u, r)
                )
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
    w_interior,
    p_attend,
    lex: ...,
    prior: ...,
    confusion: ...,
    capture_prior: ...,
    interior: ...,
    avoidable: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * (
                    L1[
                        u, r
                    ](
                        alpha,
                        w_confusion,
                        w_interior,
                        lex,
                        prior,
                        confusion,
                        interior,
                        avoidable,
                    )
                    - w_interior * at(interior, u, r)
                    - at(avoidable, u, r)
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


@memo
def L1_attend[u: UTT, r: OBJ](
    alpha,
    w_confusion,
    w_interior,
    p_attend,
    lex: ...,
    prior: ...,
    confusion: ...,
    capture_prior: ...,
    interior: ...,
    avoidable: ...,
):
    # Without depth2: L2 with its speaker reasoning about L0 instead of L1.
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * (
                    L0[u, r](lex)
                    - w_interior * at(interior, u, r)
                    - at(avoidable, u, r)
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
    n_obj = ctx.lex.shape[1]
    k_u = jnp.sum(real_lex, axis=1, keepdims=True)
    shared_fraction = (k_u - 1.0) / jnp.maximum(1.0, n_obj - 1.0)
    return jnp.sum(real_lex * shared_fraction, axis=0)


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


def compute_avoidable(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    k_u = jnp.sum(real_lex, axis=1)
    n_obj = ctx.lex.shape[1]
    ambiguity = (k_u - 1.0) / jnp.maximum(1.0, n_obj - 1.0) * (1.0 - ctx.is_sink)
    is_unique_word = (k_u == 1.0) * (1.0 - ctx.is_sink)
    has_unique = jnp.max(real_lex * is_unique_word[:, None], axis=0)
    avoidable = real_lex * ambiguity[:, None] * has_unique[None, :]
    return avoidable


def compute_interior(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    k_u = jnp.sum(real_lex, axis=1)

    fcs_matched = jnp.where(real_lex > 0, ctx.feature_count[None, :], -999.0)
    max_fc = jnp.max(fcs_matched, axis=1, keepdims=True)

    fcs_matched_min = jnp.where(real_lex > 0, ctx.feature_count[None, :], 999.0)
    min_fc = jnp.min(fcs_matched_min, axis=1, keepdims=True)

    span = max_fc - min_fc
    rel_pos = (ctx.feature_count[None, :] - min_fc) / jnp.maximum(1.0, span)

    is_eligible = (k_u[:, None] >= 3.0) & (span > 0.0) & (real_lex > 0)
    interior = 4.0 * rel_pos * (1.0 - rel_pos) * is_eligible
    return interior.astype(jnp.float32)


def build(mechanisms):
    """``(PARAMS, choice_probs)`` of the final with ``mechanisms`` on (the rest removed)."""
    on = frozenset(mechanisms)
    unknown = sorted(on - set(MECHANISM_PARAMS))
    if unknown:
        raise ValueError(f"unknown mechanisms {unknown}; known: {list(MECHANISM_PARAMS)}")
    chosen = dict(CORE_PARAMS)
    for m in MECHANISM_PARAMS:
        if m in on:
            chosen.update(MECHANISM_PARAMS[m])
    params = {name: chosen[name] for name in FINAL_ORDER if name in chosen}

    def choice_probs(p, ctx):
        return _choice(p, ctx, on)

    return params, choice_probs


def _choice(params, ctx, on):
    zero = jnp.zeros(ctx.lex.shape[1])

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

    # The prior over objects: each salience term only when its mechanism is on.
    effective_w_features = (
        (params["w_features"] if "features" in on else 0.0)
        + (ctx.valence * params["w_valence"] if "valence" in on else 0.0)
    )

    typicality = compute_typicality(ctx)

    salience_logits = (
        (params["w_singleton"] * is_singleton if "singleton" in on else zero)
        + (params["w_distinct"] * distinct if "distinct" in on else zero)
        + effective_w_features * ctx.feature_count
        + (params["w_color"] * (1.0 - ctx.grayscale) if "color" in on else zero)
        + (params["w_typical"] * typicality if "typicality" in on else zero)
    )
    prior = softmax_prior(salience_logits)

    # The speaker's costs.
    confusion = compute_confusion(ctx)
    interior = compute_interior(ctx)
    avoidable = (params["c_avoid"] if "avoidable" in on else 0.0) * compute_avoidable(ctx)
    w_interior = params["w_interior"] if "interior" in on else 0.0
    w_confusion = params["w_confusion"] if "confusion" in on else 0.0

    real_matches = ctx.lex[ctx.utterance] * (1.0 - ctx.is_sink[ctx.utterance])
    matching_count = jnp.sum(real_matches)
    is_popout = jnp.where(matching_count == 1.0, 1.0, 0.0)
    candidate_load = jnp.maximum(0.0, matching_count - 1.0)

    # The attention gate: without it the listener always attends (p_attend = 1).
    n_obj = ctx.lex.shape[1]
    if "attention" in on:
        logit_p_attend = (
            params["logit_p_base"]
            + (params["w_popout"] * is_popout if "popout" in on else 0.0)
            - (params["w_load"] * (n_obj - 2.0) if "load" in on else 0.0)
            - (params["w_candidate"] * candidate_load if "candidate" in on else 0.0)
        )
        p_attend = jax.nn.sigmoid(logit_p_attend)
    else:
        p_attend = 1.0

    # Inattention's capture by the prior over matching objects, leaking onto the others.
    inhibit_leak = (
        jax.nn.softplus(params["w_inhibit_leak"]) * candidate_load if "inhibit_leak" in on else 0.0
    )
    inhibit_factor = jnp.where(real_matches > 0, 1.0, inhibit_leak)
    capture_weights = prior * inhibit_factor + 1e-6
    capture_prior = capture_weights / jnp.sum(capture_weights)

    is_foil = compute_solitary_foil(ctx)
    solitary_foil_vec = is_foil[ctx.utterance]
    has_foil = jnp.max(solitary_foil_vec)
    p_outlier = (
        jax.nn.sigmoid(params["logit_p_outlier"]) * has_foil if "outlier" in on else 0.0
    )

    listener = L2 if "depth2" in on else L1_attend
    heard = listener(
        params["alpha"],
        w_confusion,
        w_interior,
        p_attend,
        ctx.lex,
        prior,
        confusion,
        capture_prior,
        interior,
        avoidable,
    )[ctx.utterance]
    heard_with_popout = (1.0 - p_outlier) * heard + p_outlier * solitary_foil_vec

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard_with_popout), params["lapse"]
    )
