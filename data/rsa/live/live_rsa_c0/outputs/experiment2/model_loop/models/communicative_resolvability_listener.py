"""Communicative resolvability listener: pragmatic listener reasoning about communicative resolvability.

Listeners interpret referring expressions with an a priori expectation that speakers
select communicative targets based on referential resolvability: the degree to which
an intended referent can be successfully singled out from the visual context given a
single-word communicative budget. Before hearing an informative word and during referent
selection, listeners evaluate each referent by its communicative resolvability—the maximum
descriptive recovery precision attainable across its true descriptive features—expecting
speakers to avoid intending referents that cannot be cleanly resolved in context.
Pragmatic listeners invert this resolvability-guided speaker, preventing feature-rich
competitors from overpowering minimal pragmatic targets under shared expressions,
elevating selections of full duplicates on uninformative trials, and calibrating
referent expectations across varied visual scenes.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

MODES = jnp.arange(2)

PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "logit_p_consider": dist.Normal(1.0, 1.0),
    "logit_p_base": dist.Normal(2.0, 1.0),
    "w_load": dist.Normal(0.5, 0.5),
    "w_popout": dist.Normal(1.0, 0.5),
    "w_candidate": dist.Normal(0.5, 0.5),
    "w_inhibit_leak": dist.Normal(0.0, 1.0),
    "w_resolvable": dist.Normal(1.0, 1.0),
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
    alpha, w_confusion, p_consider, lex: ..., prior: ..., confusion: ...
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            mode in MODES, wpp=p_consider if mode == 1 else (1.0 - p_consider)
        ),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * (
                exp(
                    alpha * L0[u, r](lex)
                    - w_confusion * at(confusion, u, r)
                )
                if mode == 1
                else 1.0
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L_resolvable[u: UTT, r: OBJ](
    alpha,
    w_confusion,
    p_consider,
    p_attend,
    lex: ...,
    prior: ...,
    confusion: ...,
    capture_prior: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            mode in MODES, wpp=p_consider if mode == 1 else (1.0 - p_consider)
        ),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * (
                exp(
                    alpha
                    * L1[u, r](
                        alpha, w_confusion, p_consider, lex, prior, confusion
                    )
                    - w_confusion * at(confusion, u, r)
                )
                if mode == 1
                else 1.0
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(
        att_mode in MODES, wpp=p_attend if att_mode == 1 else (1.0 - p_attend)
    )
    listener: chooses(
        r in OBJ,
        wpp=Pr[speaker.r == r] if att_mode == 1 else vec(capture_prior, r),
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


def compute_resolvability(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    k_u = jnp.sum(real_lex, axis=1, keepdims=True)
    precision = real_lex / jnp.maximum(1.0, k_u)
    return jnp.max(precision, axis=0)


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

    resolvability = compute_resolvability(ctx)

    salience_logits = (
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + effective_w_features * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
        + params["w_resolvable"] * resolvability
    )
    prior = softmax_prior(salience_logits)

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
    p_consider = jax.nn.sigmoid(params["logit_p_consider"])

    inhibit_leak = jax.nn.softplus(params["w_inhibit_leak"]) * candidate_load
    inhibit_factor = jnp.where(real_matches > 0, 1.0, inhibit_leak)
    capture_weights = prior * inhibit_factor + 1e-6
    capture_prior = capture_weights / jnp.sum(capture_weights)

    heard = L_resolvable(
        params["alpha"],
        params["w_confusion"],
        p_consider,
        p_attend,
        ctx.lex,
        prior,
        confusion,
        capture_prior,
    )[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
