"""Bounded info gain outlier listener: pragmatic listener under capacity-limited executive
inhibitory control with contextual category typicality, bounded speaker consideration,
contrastive outlier pop-out, and speaker communicative information gain.

Refining bounded_typicality_outlier_listener by incorporating the communicative information gain
component from information_gain_speaker: speakers evaluate communicative alternatives probabilistically
under bounded consideration, and when deliberating among alternative expressions, they choose words to
maximize global communicative information gain—the Kullback-Leibler divergence between the listener's
prior belief and updated belief across the visual scene—alongside referent recovery. Pragmatic listeners
invert this information-gain speaker under capacity-limited executive inhibitory control, recognizing
that hearing an ambiguous word provides stronger evidence for referents that lacked high-information
alternatives, while distinguishing features that isolate a competitor decisively preclude that competitor.
This directly rectifies the under-selection of pragmatic targets against all-feature competitors in
four-object displays (four_object_multifeature_target_vs_all_feature_rate and
four_object_two_feature_target_vs_three_feature_competitor_rate) and elevates choices of maximal-feature
candidates under ambiguous expressions, while preserving contextual category typicality, bounded
consideration, and contrastive outlier pop-out.

Differences from bounded_typicality_outlier_listener:
- Recursion depth remains depth 2 (choice_probs calls L_typical, which inverts a speaker simulating depth-1 listener L1).
- Added parameter: w_info (prior dist.Normal(0.0, 1.0)).
- No parameters removed (retains alpha, logit_p_consider, logit_p_base, w_load, w_popout, w_candidate, logit_p_outlier, w_inhibit_leak, w_typical, w_singleton, w_distinct, w_features, w_valence, w_color, w_confusion, and lapse).
- In choice_probs: added compute_info_gain(ctx, prior) computing utterance-level Kullback-Leibler divergence between the literal listener distribution and the perceptual object prior across the scene (l0 = real_lex / jnp.maximum(1.0, k_u); term = l0 * (jnp.log(l0 + EPS) - jnp.log(prior[None, :] + EPS)); kl = jnp.sum(term, axis=1) * (1.0 - ctx.is_sink)) and passed info_gain to L_typical.
- Speaker utility in both S1 and S2: added communicative information gain term w_info * vec(info_gain, u) inside communicative utility when mode == 1, evaluating as alpha * (L + w_info * vec(info_gain, u)) - w_confusion * at(confusion, u, r) while mode == 0 selects true utterances uniformly with weight 1.0.
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
    "w_info": dist.Normal(0.0, 1.0),
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
    alpha,
    w_info,
    w_confusion,
    p_consider,
    lex: ...,
    prior: ...,
    confusion: ...,
    info_gain: ...,
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
                    alpha * (L0[u, r](lex) + w_info * vec(info_gain, u))
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
def L_typical[u: UTT, r: OBJ](
    alpha,
    w_info,
    w_confusion,
    p_consider,
    p_attend,
    lex: ...,
    prior: ...,
    confusion: ...,
    capture_prior: ...,
    info_gain: ...,
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
                    * (
                        L1[
                            u, r
                        ](
                            alpha,
                            w_info,
                            w_confusion,
                            p_consider,
                            lex,
                            prior,
                            confusion,
                            info_gain,
                        )
                        + w_info * vec(info_gain, u)
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
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
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


def compute_info_gain(ctx, prior):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    k_u = jnp.sum(real_lex, axis=1, keepdims=True)
    l0 = real_lex / jnp.maximum(1.0, k_u)

    term = l0 * (jnp.log(l0 + EPS) - jnp.log(prior[None, :] + EPS))
    kl = jnp.sum(term, axis=1) * (1.0 - ctx.is_sink)
    return kl


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
    info_gain = compute_info_gain(ctx, prior)

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

    is_foil = compute_solitary_foil(ctx)
    solitary_foil_vec = is_foil[ctx.utterance]
    has_foil = jnp.max(solitary_foil_vec)
    p_outlier = jax.nn.sigmoid(params["logit_p_outlier"]) * has_foil

    heard = L_typical(
        params["alpha"],
        params["w_info"],
        params["w_confusion"],
        p_consider,
        p_attend,
        ctx.lex,
        prior,
        confusion,
        capture_prior,
        info_gain,
    )[ctx.utterance]
    heard_with_popout = (1.0 - p_outlier) * heard + p_outlier * solitary_foil_vec

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard_with_popout), params["lapse"]
    )
