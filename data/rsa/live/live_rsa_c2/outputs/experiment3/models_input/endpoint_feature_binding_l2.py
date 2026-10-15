import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "kappa": dist.LogNormal(0.0, 1.0),
    "w_binding": dist.Beta(1.0, 24.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_empty": dist.Normal(0.0, 1.0),
    "w_endpoint": dist.Normal(0.0, 1.0),
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
def S1[u: UTT, r: OBJ](alpha, lex: ..., alt_weight: ..., accessibility: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r)
        * vec(alt_weight, u)
        * vec(accessibility, u)
        * exp(alpha * log(L0[u, r](lex) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L1[u: UTT, r: OBJ](kappa, lex: ..., prior: ..., s1: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * vec(prior, r)
        * exp(kappa * log(at(s1, u, r) + {EPS})),
    )
    return Pr[listener.r == r]


@memo
def S2[u: UTT, r: OBJ](alpha, lex: ..., l1: ..., alt_weight: ..., accessibility: ...):
    speaker: knows(r)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r)
        * vec(alt_weight, u)
        * vec(accessibility, u)
        * exp(alpha * log(at(l1, u, r) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L2[u: UTT, r: OBJ](kappa, lex: ..., prior: ..., s2: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * vec(prior, r)
        * exp(kappa * log(at(s2, u, r) + {EPS})),
    )
    return Pr[listener.r == r]


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

    k_min = jnp.min(ctx.feature_count)
    k_max = jnp.max(ctx.feature_count)
    is_inter = jnp.where(
        (ctx.feature_count > k_min) & (ctx.feature_count < k_max), 1.0, 0.0
    )
    is_endpoint = 1.0 - is_inter

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_empty"] * is_empty
        + params["w_endpoint"] * is_endpoint
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

    s1 = S1(params["alpha"], ctx.lex, alt_weight, accessibility)
    l1 = L1(params["kappa"], ctx.lex, prior, s1)
    s2 = S2(params["alpha"], ctx.lex, l1, alt_weight, accessibility)
    heard = L2(params["kappa"], ctx.lex, prior, s2)[ctx.utterance]

    # Feature Integration Theory: illusory feature binding on non-matching distractors
    is_false = (1.0 - ctx.is_sink[ctx.utterance]) * (1.0 - ctx.lex[ctx.utterance])
    max_fc = jnp.maximum(jnp.max(ctx.feature_count), 1.0)
    # Illusory binding susceptibility scales with the distractor's feature complexity
    binding_weight = is_false * (ctx.feature_count / max_fc) * prior
    total_binding = jnp.sum(binding_weight)
    p_bind = binding_weight / jnp.maximum(total_binding, EPS)
    has_binding = jnp.where(total_binding > 0, 1.0, 0.0)
    effective_bind = params["w_binding"] * has_binding

    p_word = (1.0 - effective_bind) * heard + effective_bind * p_bind
    p_target = jnp.where(ctx.is_prior > 0, prior, p_word)

    return with_lapse(p_target, params["lapse"])
