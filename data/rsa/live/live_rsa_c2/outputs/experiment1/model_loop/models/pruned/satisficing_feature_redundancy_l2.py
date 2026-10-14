"""RSA pragmatic listener at depth 2 with feature redundancy discounting and satisficing speaker informativeness.

Refining feature_redundancy_discounted_l2 by incorporating satisficing speaker informativeness from
satisficing_speaker_rsa: speakers in reference games do not linearly optimize informativeness to perfection;
rather, they are boundedly rational satisficers who evaluate candidate utterances against an aspiration
threshold theta. Once a true referring expression achieves an informativeness level meeting the aspiration
threshold, it provides diminishing marginal returns for additional uniqueness. This prevents the artificial
collapse of competitor probability across scalar reference games, naturally explaining why listeners choose
overspecified competitors and duplicate competitors substantially more often than predicted by linear RSA.
Furthermore, speakers discount redundant communicative alternatives according to contextual Jaccard overlap,
preventing correlated distinguishing features from repeatedly multiplying the preemption penalty. Depth-2
listeners invert this satisficing, redundancy-discounted speaker while retaining common knowledge of the visual
object prior (singleton salience, distinctiveness, visual feature count, and color contrast) and unbiased
literal semantics at L0. On uninformative prior trials, choices follow the integrated object prior. A lapse
parameter mixes in uniform guessing.

Differences from feature_redundancy_discounted_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: theta (prior: dist.Beta(2.0, 2.0)).
- Parameters removed: None.
- Speaker utility: Replaces linear communicative informativeness log(L + EPS) with satisficing utility
  log((L / (L + theta)) + EPS) in speaker choices at depth 1 (L1) and depth 2 (L2).
- All other terms: Contextual feature redundancy discounting on speaker alternatives (w_redundancy),
  pairwise Jaccard overlap calculation, visual referent prior components (w_singleton, w_distinct,
  w_features, w_color), unbiased literal semantics at L0, depth-2 RSA recursion (L1, L2), and uniform
  lapse mixing remain identical to the source model.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "theta": dist.Beta(2.0, 2.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_redundancy": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_redundancy(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Compute contextual redundancy for each utterance via pairwise Jaccard overlap."""
    real_lex = lex * (1.0 - is_sink)[:, None]
    intersection = jnp.dot(real_lex, real_lex.T)
    counts = jnp.sum(real_lex, axis=1)
    union = counts[:, None] + counts[None, :] - intersection
    jaccard = jnp.where(union > 0, intersection / (union + 1e-10), 0.0)
    jaccard = jnp.where(jnp.eye(lex.shape[0]) > 0, 1.0, jaccard)
    sink_mask = (
        is_sink[:, None] * (1.0 - is_sink[None, :])
        + (1.0 - is_sink[:, None]) * is_sink[None, :]
    )
    jaccard = jnp.where(sink_mask > 0, 0.0, jaccard)
    return jnp.sum(jaccard, axis=1)


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, theta, lex: ..., prior: ..., alt_weight: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * vec(alt_weight, u)
            * exp(
                alpha
                * log(
                    (L0[u, r](lex) / (L0[u, r](lex) + theta))
                    + {EPS}
                )
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, theta, lex: ..., prior: ..., alt_weight: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * vec(alt_weight, u)
            * exp(
                alpha
                * log(
                    (L1[u, r](alpha, theta, lex, prior, alt_weight)
                     / (L1[u, r](alpha, theta, lex, prior, alt_weight) + theta))
                    + {EPS}
                )
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
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

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    redundancy = compute_redundancy(ctx.lex, ctx.is_sink)
    alt_weight = jnp.exp(
        -params["w_redundancy"] * jnp.log(jnp.maximum(redundancy, 1.0))
    )

    heard = L2(params["alpha"], params["theta"], ctx.lex, prior, alt_weight)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
