"""RSA pragmatic listener at depth 2 with visual singleton salience, contextual feature distinctiveness, competitor confusion avoidance, and omitted-alternative exhaustification penalty.

Refining rsa_l2_singleton_confusion_omission by incorporating contextual feature distinctiveness from rsa_l2_salience_confusion_l0:
Visual salience in reference games operates continuously across feature space rather than solely as an all-or-nothing binary distinction between duplicates and non-duplicates.
Listeners and simulated speakers evaluate candidate referents with common-ground visual priors that combine discrete singleton salience with continuous contextual distinctiveness (the accumulated visual feature contrast between each object and all other display items), directing attention toward contextually distinctive objects even when all items in the display are distinct.
When multiple descriptions apply to an object, speakers avoid using an ambiguous descriptor if the target possesses an unmentioned, highly discriminative alternative feature in the display, and avoid words that create visual confusion with competitors.
Pragmatic listeners invert this multi-criterion speaker at depth 2.
On uninformative prior trials, choices follow the combined singleton salience and distinctiveness prior directly.
A lapse parameter mixes in uniform guessing.

Differences from rsa_l2_singleton_confusion_omission:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: w_distinct (Normal(0.0, 1.0), weighting the continuous contextual feature distinctiveness prior over candidate referents).
- Parameters removed: None.
- Other terms: Added compute_distinctiveness to calculate the accumulated visual feature distance between each referent and all other display objects; incorporated + params["w_distinct"] * distinct into the common-ground softmax object prior passed to simulated speakers S1 and S2 and governing choices on uninformative prior trials.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 1.0),
    "w_omission": dist.Normal(0.0, 1.0),
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
    lex: ...,
    prior: ...,
    confusion: ...,
    violation: ...,
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
    lex: ...,
    prior: ...,
    confusion: ...,
    violation: ...,
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
                        lex,
                        prior,
                        confusion,
                        violation,
                    )
                    + {EPS}
                )
                - w_confusion * at(confusion, u, r)
                - w_omission * at(violation, u, r)
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


def compute_distinctiveness(ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    return jnp.sum(diff, axis=(0, 2))


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

    distinct = compute_distinctiveness(ctx)
    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
    )
    confusion = compute_confusion(ctx)
    violation = compute_exhaustification_violation(ctx)

    heard = L2(
        params["alpha"],
        params["w_confusion"],
        params["w_omission"],
        ctx.lex,
        prior,
        confusion,
        violation,
    )[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
