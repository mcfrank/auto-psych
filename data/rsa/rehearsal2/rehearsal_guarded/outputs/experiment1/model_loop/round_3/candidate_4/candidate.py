"""RSA pragmatic listener at depth 2 with visual singleton salience, competitor confusion avoidance, and utterance extension cost.

Refines rsa_l2_singleton_confusion by incorporating utterance extension costs from isolated_graded_costly_l3:
speakers and listeners reason at depth 2 with common knowledge of visual singleton salience. In addition,
simulated speakers evaluate candidate words by balancing communicative informativeness against both competitor
confusion avoidance (penalizing words whose competitors share many visual features) and utterance extension cost
(penalizing words proportional to the fraction of objects in the scene sharing the feature). When describing an object
that possesses multiple valid features, speakers prefer more specific, less widely shared descriptors, guiding listeners
to reject referents for which the expression was non-specific.
On uninformative prior trials, choices follow the visual singleton salience prior directly.
A lapse parameter mixes in uniform guessing.

Differences from rsa_l2_singleton_confusion:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: w_cost (Normal(0.0, 1.0), penalizing candidate words proportional to their visual extension / scene frequency).
- Parameters removed: None.
- Other terms: Added compute_extension_cost to calculate the scene-wide object fraction sharing each word;
  incorporated - w_cost * vec(cost, u) into simulated speaker utility at both depth-1 (S1 in L1) and depth-2 (S2 in L2).
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 1.0),
    "w_cost": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_confusion, w_cost, lex: ..., prior: ..., confusion: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - w_confusion * at(confusion, u, r)
                - w_cost * vec(cost, u)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, w_confusion, w_cost, lex: ..., prior: ..., confusion: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, w_confusion, w_cost, lex, prior, confusion, cost) + {EPS})
                - w_confusion * at(confusion, u, r)
                - w_cost * vec(cost, u)
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

    confusion = real_lex @ sim
    return confusion


def compute_extension_cost(ctx):
    real_words = 1.0 - ctx.is_sink
    n_obj = ctx.lex.shape[1]
    n_true = jnp.sum(ctx.lex * real_words[:, None], axis=1)
    return real_words * (n_true / n_obj)


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

    prior = softmax_prior(params["w_singleton"] * is_singleton)
    confusion = compute_confusion(ctx)
    cost = compute_extension_cost(ctx)

    heard = L2(
        params["alpha"],
        params["w_confusion"],
        params["w_cost"],
        ctx.lex,
        prior,
        confusion,
        cost,
    )[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
