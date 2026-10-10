"""RSA pragmatic listener at depth 2 with visual singleton salience, competitor confusion avoidance, omitted-alternative exhaustification penalty, utterance extension cost, and exclusive alternative penalty.

Refining rsa_l2_singleton_confusion_omission_cost by incorporating exclusive alternative penalties from exclusive_alternative_speaker:
speakers and listeners reason at depth 2 with common knowledge of visual singleton salience.
Simulated speakers evaluate candidate referring expressions not only by informativeness to simulated listeners,
distractor confusion avoidance, omitted-alternative exhaustification penalties, and utterance extension costs,
but also by penalizing ambiguous descriptors whenever the intended referent possesses a foolproof exclusive descriptor in the display.
When an object possesses an exclusive descriptor that uniquely singles it out in the scene, speakers avoid choosing
an ambiguous shared expression, guiding listeners to direct ambiguous expressions toward candidate referents that lack any exclusive descriptor.
On uninformative prior trials, choices follow the singleton salience prior directly.
A lapse parameter mixes in uniform guessing.

Differences from rsa_l2_singleton_confusion_omission_cost:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: w_exclusive (Normal(0.0, 1.0), penalizing ambiguous referring expressions when the intended referent possesses at least one uniquely distinguishing descriptor in the visual context).
- Parameters removed: None.
- Other terms: Added compute_exclusive_penalty to identify referring expressions that are non-exclusive for referents possessing at least one contextually unique descriptor;
  incorporated - w_exclusive * at(exclusive_penalty, u, r) into simulated speaker utility at both depth-1 (S1 in L1) and depth-2 (S2 in L2).
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 1.0),
    "w_omission": dist.Normal(0.0, 1.0),
    "w_cost": dist.Normal(0.0, 1.0),
    "w_exclusive": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[
    u: UTT,
    r: OBJ,
](
    alpha,
    w_confusion,
    w_omission,
    w_cost,
    w_exclusive,
    lex: ...,
    prior: ...,
    confusion: ...,
    violation: ...,
    cost: ...,
    exclusive_penalty: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - w_confusion * at(confusion, u, r)
                - w_omission * at(violation, u, r)
                - w_cost * vec(cost, u)
                - w_exclusive * at(exclusive_penalty, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[
    u: UTT,
    r: OBJ,
](
    alpha,
    w_confusion,
    w_omission,
    w_cost,
    w_exclusive,
    lex: ...,
    prior: ...,
    confusion: ...,
    violation: ...,
    cost: ...,
    exclusive_penalty: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * log(
                    L1[u, r](
                        alpha,
                        w_confusion,
                        w_omission,
                        w_cost,
                        w_exclusive,
                        lex,
                        prior,
                        confusion,
                        violation,
                        cost,
                        exclusive_penalty,
                    )
                    + {EPS}
                )
                - w_confusion * at(confusion, u, r)
                - w_omission * at(violation, u, r)
                - w_cost * vec(cost, u)
                - w_exclusive * at(exclusive_penalty, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
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

    confusion = real_lex @ sim
    return confusion


def compute_exhaustification_violation(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words

    n_obj_per_feat = jnp.sum(real_lex, axis=1)
    info = jnp.where(n_obj_per_feat > 0, 1.0 / n_obj_per_feat, 0.0)

    total_obj_info = jnp.sum(real_lex * info[:, None], axis=0)
    violation = total_obj_info[None, :] - real_lex * info[:, None]
    return jnp.maximum(violation, 0.0)


def compute_extension_cost(ctx):
    real_words = 1.0 - ctx.is_sink
    n_obj = ctx.lex.shape[1]
    n_true = jnp.sum(ctx.lex * real_words[:, None], axis=1)
    return real_words * (n_true / n_obj)


def compute_exclusive_penalty(ctx):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]

    n_obj_per_feat = jnp.sum(real_lex, axis=1)
    is_exclusive_feat = jnp.where(n_obj_per_feat == 1, 1.0, 0.0) * real_words

    obj_has_exclusive = jnp.sum(real_lex * is_exclusive_feat[:, None], axis=0)
    has_exclusive = jnp.where(obj_has_exclusive > 0, 1.0, 0.0)

    penalty = has_exclusive[None, :] * (1.0 - is_exclusive_feat[:, None]) * real_lex
    return penalty


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
    violation = compute_exhaustification_violation(ctx)
    cost = compute_extension_cost(ctx)
    exclusive_penalty = compute_exclusive_penalty(ctx)

    heard = L2(
        params["alpha"],
        params["w_confusion"],
        params["w_omission"],
        params["w_cost"],
        params["w_exclusive"],
        ctx.lex,
        prior,
        confusion,
        violation,
        cost,
        exclusive_penalty,
    )[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
