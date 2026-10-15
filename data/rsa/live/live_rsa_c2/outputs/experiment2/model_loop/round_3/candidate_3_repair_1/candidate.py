"""Aspect-informative speaker depth-2 pragmatic listener.

When communicating in visual reference games, speakers do not exclusively pursue
referent identification (the identity Question Under Discussion: "which object is it?").
Speakers also pursue an aspect-oriented goal (an aspect QUD: "what property does this
object have?"), selecting utterances that are informative about a specific attribute
aspect of the referent. Under an aspect goal, the speaker evaluates how well an
utterance conveys an applicable feature of the object rather than seeking to contrast
the object against context distractors, producing applicable labels for multi-featured
objects even when unmentioned distinguishing features are present. Pragmatic listeners
reason joint-inferentially over the speaker's communicative goal and intended referent
across recursive Theory of Mind tiers, softening scalar preemption penalties against
multi-featured competitor referents while preserving target identification when an
identity goal is inferred.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "kappa": dist.LogNormal(0.0, 1.0),
    "w_aspect": dist.Beta(1.0, 4.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_empty": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_redundancy": dist.Normal(0.0, 1.0),
    "w_prevalence": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_equivalence_class(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Compute the referential equivalence class size for each utterance."""
    real_lex = lex * (1.0 - is_sink)[:, None]
    diff = jnp.abs(real_lex[:, None, :] - real_lex[None, :, :])
    same_ext = jnp.all(diff == 0, axis=-1).astype(jnp.float32)
    real_mask = (1.0 - is_sink[:, None]) * (1.0 - is_sink[None, :])
    same_ext = same_ext * real_mask
    k_u = jnp.sum(same_ext, axis=1)
    return jnp.where(is_sink > 0, 1.0, jnp.maximum(k_u, 1.0))


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def S_ident[u: UTT, r: OBJ](alpha, lex: ..., l_prev: ..., alt_weight: ..., accessibility: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r)
        * vec(alt_weight, u)
        * vec(accessibility, u)
        * exp(alpha * log(at(l_prev, u, r) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L_tier[u: UTT, r: OBJ](kappa, lex: ..., prior: ..., s: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * vec(prior, r)
        * exp(kappa * log(at(s, u, r) + {EPS})),
    )
    return Pr[listener.r == r]


def compute_aspect_speaker(
    alpha: float,
    l_prev: jnp.ndarray,
    lex: jnp.ndarray,
    accessibility: jnp.ndarray,
    alt_weight: jnp.ndarray,
) -> jnp.ndarray:
    """Compute speaker utterance distribution under an aspect QUD."""
    # M[u, f]: probability that listener hearing u attributes aspect f to the object
    M = jnp.matmul(l_prev, lex.T)

    base_u = lex * alt_weight[:, None] * accessibility[:, None]
    w_u_f_r = base_u[:, None, :] * jnp.exp(alpha * jnp.log(M[:, :, None] + EPS))
    p_u_given_r_f = w_u_f_r / jnp.maximum(jnp.sum(w_u_f_r, axis=0, keepdims=True), EPS)

    p_f_given_r = base_u / jnp.maximum(jnp.sum(base_u, axis=0, keepdims=True), EPS)
    s_aspect = jnp.sum(p_u_given_r_f * p_f_given_r[None, :, :], axis=1)
    return s_aspect


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

    is_empty = jnp.where(ctx.feature_count == 0, 1.0, 0.0)

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_empty"] * is_empty
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    k_u = compute_equivalence_class(ctx.lex, ctx.is_sink)
    alt_weight = jnp.exp(-params["w_redundancy"] * jnp.log(k_u))

    freq = jnp.sum(real_lex, axis=1)
    accessibility = jnp.where(
        ctx.is_sink > 0,
        1.0,
        jnp.exp(params["w_prevalence"] * jnp.log(jnp.maximum(freq, 1.0))),
    )

    # Level 0
    l0 = L0(ctx.lex)

    # Level 1
    s1_ident = S_ident(params["alpha"], ctx.lex, l0, alt_weight, accessibility)
    s1_aspect = compute_aspect_speaker(
        params["alpha"], l0, ctx.lex, accessibility, alt_weight
    )
    s1_joint = (1.0 - params["w_aspect"]) * s1_ident + params["w_aspect"] * s1_aspect
    l1 = L_tier(params["kappa"], ctx.lex, prior, s1_joint)

    # Level 2
    s2_ident = S_ident(params["alpha"], ctx.lex, l1, alt_weight, accessibility)
    s2_aspect = compute_aspect_speaker(
        params["alpha"], l1, ctx.lex, accessibility, alt_weight
    )
    s2_joint = (1.0 - params["w_aspect"]) * s2_ident + params["w_aspect"] * s2_aspect
    heard = L_tier(params["kappa"], ctx.lex, prior, s2_joint)[ctx.utterance]

    # Attentional lapse captured by perceptual prior
    p_choice = (1.0 - params["lapse"]) * heard + params["lapse"] * prior
    return jnp.where(ctx.is_prior > 0, prior, p_choice)
