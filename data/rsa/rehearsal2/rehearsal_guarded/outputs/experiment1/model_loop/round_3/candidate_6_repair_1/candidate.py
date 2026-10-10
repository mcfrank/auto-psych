"""RSA pragmatic listener at depth 2 with singleton salience, competitor confusion avoidance, omitted-alternative exhaustification penalty, and a softmax decision rule.

Refining rsa_l2_singleton_confusion_omission by incorporating a softmax decision rule (choice determinism) from softmax_belief_listener:
listeners evaluate candidate referents via depth-2 pragmatic reasoning with common knowledge of visual singleton salience,
competitor confusion avoidance, and omitted-alternative exhaustification, but convert their posterior beliefs into choices
through a power-law softmax decision rule rather than strict probability matching. A decision determinism parameter gamma
sharpens selections toward the highest-probability referent. On uninformative trials, choices default to the visual singleton
salience prior. A lapse parameter mixes in uniform guessing.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 0.5),
    "w_singleton": dist.Normal(0.0, 1.0),
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
def L1[u: UTT, r: OBJ](alpha, w_confusion, w_omission, lex: ..., prior: ..., confusion: ..., violation: ...):
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
def L2[u: UTT, r: OBJ](alpha, w_confusion, w_omission, lex: ..., prior: ..., confusion: ..., violation: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, w_confusion, w_omission, lex, prior, confusion, violation) + {EPS})
                - w_confusion * at(confusion, u, r)
                - w_omission * at(violation, u, r)
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


def compute_exhaustification_violation(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words

    n_obj_per_feat = jnp.sum(real_lex, axis=1)
    info = jnp.where(n_obj_per_feat > 0, 1.0 / n_obj_per_feat, 0.0)

    total_obj_info = jnp.sum(real_lex * info[:, None], axis=0)
    violation = total_obj_info[None, :] - real_lex * info[:, None]
    return jnp.maximum(violation, 0.0)


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

    heard = L2(
        params["alpha"],
        params["w_confusion"],
        params["w_omission"],
        ctx.lex,
        prior,
        confusion,
        violation,
    )[ctx.utterance]

    logits = params["gamma"] * jnp.log(heard + EPS)
    dec = jax.nn.softmax(logits)

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, dec), params["lapse"]
    )
