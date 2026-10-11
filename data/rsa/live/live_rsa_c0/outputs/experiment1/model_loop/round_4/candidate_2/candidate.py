"""Feature binding listener: pragmatic listener under capacity-limited visual feature binding.

Listeners interpret referring expressions under capacity-limited visual feature binding
(Treisman's Feature Integration Theory), where visual features can undergo illusory
conjunction and migrate to neighboring objects with probability proportional to their
scene-wide prevalence. When hearing a word that applies to almost all objects in the display,
the high contextual prevalence of that feature causes occasional perceptual migration to
the solitary foil, resolving the under-prediction of non-matching foil choices while
preserving literal precision on pop-out displays. Across larger displays and redundant
feature overlaps, this prevalence-dependent perceptual scaling naturally softens scalar
implicatures and elevates choices of feature-rich referents.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "logit_beta_bind": dist.Normal(-2.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](bound_lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(bound_lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, bound_lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(bound_lex, u, r) * exp(alpha * log(L0[u, r](bound_lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, bound_lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(bound_lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, bound_lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_bound_lex(lex, is_sink, beta_bind):
    real_words = (1.0 - is_sink)[:, None]
    real_lex = lex * real_words
    n_obj = lex.shape[1]
    prevalence = jnp.sum(real_lex, axis=1, keepdims=True) / n_obj
    bound = (1.0 - beta_bind) * lex + beta_bind * prevalence
    return jnp.where(is_sink[:, None] > 0, lex, bound)


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

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + effective_w_features * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    beta_bind = jax.nn.sigmoid(params["logit_beta_bind"])
    bound_lex = compute_bound_lex(ctx.lex, ctx.is_sink, beta_bind)

    heard = L2(params["alpha"], bound_lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
