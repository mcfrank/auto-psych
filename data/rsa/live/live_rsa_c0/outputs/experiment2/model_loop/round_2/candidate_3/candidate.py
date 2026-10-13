"""Lexical feature uncertainty listener: pragmatic listener reasoning under uncertainty about which visual feature a speaker's word denotes.

Listeners interpret referring expressions under lexical feature uncertainty: rather than assuming an
error-free mapping from words to visual features, listeners are uncertain which contextual feature
dimension a speaker's word denotes. When hearing a word, listeners treat it as primarily denoting
its literal feature with probability 1 - p_uncertain, but consider with probability p_uncertain that
it denotes other contextual visual features present in the scene. Because feature-rich referents
instantiate more visual features that could be intended under feature uncertainty, they receive
elevated semantic applicability, softening excessive scalar exclusion, explaining preferences for
maximal-feature objects on ambiguous displays, and naturally accounting for non-matching distractor intrusions.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "logit_p_uncertain": dist.Normal(-2.5, 1.0),
    "w_confusion": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., prior: ..., confusion: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
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
def L2[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., prior: ..., confusion: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * L1[u, r](alpha, w_confusion, lex, prior, confusion)
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
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

    salience_logits = (
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + effective_w_features * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
    )
    prior = softmax_prior(salience_logits)
    confusion = compute_confusion(ctx)

    l2 = L2(params["alpha"], params["w_confusion"], ctx.lex, prior, confusion)

    # Contextual lexical feature uncertainty:
    # When hearing utterance u, the listener considers that u primarily denotes feature u
    # with probability 1 - p_uncertain, and with probability p_uncertain denotes another
    # contextual visual feature in the scene.
    p_uncertain = jax.nn.sigmoid(params["logit_p_uncertain"])
    real_mask = 1.0 - ctx.is_sink
    n_real = jnp.sum(real_mask)
    n_utt = ctx.lex.shape[0]

    is_heard = jnp.arange(n_utt) == ctx.utterance
    is_other_real = real_mask * (1.0 - is_heard)
    denom = jnp.maximum(1.0, n_real - 1.0)
    other_weight = (p_uncertain / denom) * is_other_real
    heard_weight = (1.0 - p_uncertain) * is_heard

    weights = jnp.where(
        n_real <= 1.0,
        jnp.where(is_heard, 1.0, 0.0),
        heard_weight + other_weight,
    )
    weights = weights / jnp.maximum(1e-6, jnp.sum(weights))

    # Marginalize over uncertain feature denotations:
    heard = jnp.sum(weights[:, None] * l2, axis=0)

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
