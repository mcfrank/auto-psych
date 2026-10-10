"""RSA pragmatic listener at depth 2 with dilemma fallback choice rationality, visual singleton salience, competitor confusion avoidance, omitted-alternative exhaustification penalty, and utterance extension cost.

Refining rsa_l2_singleton_confusion_omission_cost by incorporating dilemma fallback choice rationality from dilemma_fallback_speaker:
When an intended referent can be uniquely distinguished in the visual context, speakers produce distinguishing expressions with communicative rationality alpha.
When an intended referent cannot be uniquely named in the display, the speaker faces an expressive dilemma where no word guarantees recovery and selects among imperfect descriptors with a distinct dilemma choice rationality alpha_dilemma.
Pragmatic listeners at depth 2 invert this two-regime speaker, recognizing that ambiguous expressions indicate referents in expressive dilemmas rather than referents whose unique descriptors were withheld.
On uninformative prior trials, choices follow the singleton salience prior directly.
A lapse parameter captures random clicking.

Differences from rsa_l2_singleton_confusion_omission_cost:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: alpha_dilemma (LogNormal(0.0, 1.0), governing speaker choice rationality when the intended referent has no uniquely distinguishing descriptor).
- Parameters removed: None.
- Other terms: Added compute_nameability to identify referents that possess at least one uniquely distinguishing descriptor in the visual context; incorporated vec(is_nameable, r) * alpha + (1.0 - vec(is_nameable, r)) * alpha_dilemma into simulated speaker utility at both depth-1 (S1 in L1) and depth-2 (S2 in L2).
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "alpha_dilemma": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 1.0),
    "w_omission": dist.Normal(0.0, 1.0),
    "w_cost": dist.Normal(0.0, 1.0),
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
    alpha_dilemma,
    w_confusion,
    w_omission,
    w_cost,
    lex: ...,
    prior: ...,
    confusion: ...,
    violation: ...,
    cost: ...,
    is_nameable: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                (
                    vec(is_nameable, r) * alpha
                    + (1.0 - vec(is_nameable, r)) * alpha_dilemma
                )
                * log(L0[u, r](lex) + {EPS})
                - w_confusion * at(confusion, u, r)
                - w_omission * at(violation, u, r)
                - w_cost * vec(cost, u)
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
    alpha_dilemma,
    w_confusion,
    w_omission,
    w_cost,
    lex: ...,
    prior: ...,
    confusion: ...,
    violation: ...,
    cost: ...,
    is_nameable: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                (
                    vec(is_nameable, r) * alpha
                    + (1.0 - vec(is_nameable, r)) * alpha_dilemma
                )
                * log(
                    L1[u, r](
                        alpha,
                        alpha_dilemma,
                        w_confusion,
                        w_omission,
                        w_cost,
                        lex,
                        prior,
                        confusion,
                        violation,
                        cost,
                        is_nameable,
                    )
                    + {EPS}
                )
                - w_confusion * at(confusion, u, r)
                - w_omission * at(violation, u, r)
                - w_cost * vec(cost, u)
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


def compute_nameability(ctx):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    n_true = jnp.sum(real_lex, axis=1)

    is_unique_utt = jnp.where((n_true == 1) & (real_words > 0), 1.0, 0.0)
    has_unique = jnp.sum(real_lex * is_unique_utt[:, None], axis=0)
    return jnp.where(has_unique > 0, 1.0, 0.0)


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
    is_nameable = compute_nameability(ctx)

    heard = L2(
        params["alpha"],
        params["alpha_dilemma"],
        params["w_confusion"],
        params["w_omission"],
        params["w_cost"],
        ctx.lex,
        prior,
        confusion,
        violation,
        cost,
        is_nameable,
    )[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
