"""RSA pragmatic listener at depth 2 with attentional crowding, singleton isolation, competitor confusion aversion, and chromatic prominence.

Refines crowding_singleton_confusion_l2 by incorporating chromatic prominence
(taking the color salience component from crowding_singleton_chromatic_l2)
into the shared referent prior: listeners and the speakers they model treat
colorful objects as perceptually salient referents over desaturated grayscale
items, alongside unique singletons, attentional crowding penalties for shared
feature overlap, and competitor confusion aversion.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 2.0),
    "w_crowd": dist.Normal(0.0, 2.0),
    "w_confusion": dist.Normal(0.0, 2.0),
    "w_color": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., prior: ..., confusion: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex, prior) + {EPS})
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., prior: ..., confusion: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, w_confusion, lex, prior, confusion) + {EPS})
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_prior_and_confusion(ctx, w_singleton, w_crowd, w_color):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    is_identical = (dist < 1e-5).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_singleton = (copy_count <= 1.0).astype(jnp.float32)

    n_obj = dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)
    crowding = jnp.sum(sim, axis=1)

    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    confusion = real_lex @ sim

    color_prominence = 1.0 - ctx.grayscale

    prior = softmax_prior(
        w_singleton * is_singleton - w_crowd * crowding + w_color * color_prominence
    )
    return prior, confusion


def choice_probs(params, ctx):
    prior, confusion = compute_prior_and_confusion(
        ctx, params["w_singleton"], params["w_crowd"], params["w_color"]
    )
    heard = L2(
        params["alpha"],
        params["w_confusion"],
        ctx.lex,
        prior,
        confusion,
    )[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
