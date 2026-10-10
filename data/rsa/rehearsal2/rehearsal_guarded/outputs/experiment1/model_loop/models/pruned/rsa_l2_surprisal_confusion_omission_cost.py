"""RSA pragmatic listener at depth 2 with visual singleton salience, Shannon feature surprisal, competitor confusion avoidance, omitted-alternative exhaustification penalty, and utterance extension cost.

Refining rsa_l2_singleton_confusion_omission_cost by incorporating feature surprisal salience from feature_surprisal_salience_listener:
Speakers and listeners reason at depth 2 with common knowledge of an object prior combining discrete visual singleton salience (favoring unique objects lacking identical duplicates in the scene) with continuous Shannon feature surprisal (favoring objects that possess rare, unexpected visual features in the context).
Simulated speakers evaluate candidate referring expressions by balancing informativeness against competitor confusion avoidance, omitted-alternative exhaustification penalties, and utterance extension costs proportional to word generality.
On uninformative prior trials, choices follow the combined singleton salience and feature surprisal prior directly.
A lapse parameter mixes in uniform guessing.

Differences from rsa_l2_singleton_confusion_omission_cost:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: w_surprisal (Normal(0.0, 1.0), weighting the Shannon feature surprisal prior over candidate referents).
- Parameters removed: None.
- Other terms: Added compute_feature_surprisal to calculate the accumulated Shannon self-information of each object's features relative to their scene frequencies;
  incorporated + params["w_surprisal"] * obj_surprisal into the softmax object prior passed to simulated speakers S1 and S2 and governing choices on uninformative prior trials.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_surprisal": dist.Normal(0.0, 1.0),
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
    w_confusion,
    w_omission,
    w_cost,
    lex: ...,
    prior: ...,
    confusion: ...,
    violation: ...,
    cost: ...,
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
    lex: ...,
    prior: ...,
    confusion: ...,
    violation: ...,
    cost: ...,
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
                        lex,
                        prior,
                        confusion,
                        violation,
                        cost,
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


def compute_feature_surprisal(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_obj = ctx.lex.shape[1]
    n_r = jnp.sum(real_lex, axis=1)
    p_u = jnp.where(n_r > 0, n_r / n_obj, 1.0)
    surprisal = jnp.where(n_r > 0, -jnp.log(p_u), 0.0)
    obj_surprisal = jnp.sum(real_lex * surprisal[:, None], axis=0)
    return obj_surprisal


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

    obj_surprisal = compute_feature_surprisal(ctx)

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_surprisal"] * obj_surprisal
    )
    confusion = compute_confusion(ctx)
    violation = compute_exhaustification_violation(ctx)
    cost = compute_extension_cost(ctx)

    heard = L2(
        params["alpha"],
        params["w_confusion"],
        params["w_omission"],
        params["w_cost"],
        ctx.lex,
        prior,
        confusion,
        violation,
        cost,
    )[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
