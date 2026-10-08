"""RSA pragmatic listener at depth 2 with singleton isolation, feature parsimony, and attentional crowding.

Refines singleton_parsimony_l2 by incorporating continuous visual attentional
crowding (taking the visual crowding component from attentional_crowding_listener)
into the shared referent prior. Listeners and the speakers they model combine
perceptual isolation for contextually unique singletons lacking identical clones
and an intrinsic preference for feature-sparse minimal referents with an
attentional penalty for visual crowding caused by shared feature overlap with
other display items. This shared prior is common knowledge across all levels of
recursion (L0, S1, L1, S2, L2).
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 2.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_crowd": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_prior(ctx, w_singleton, w_features, w_crowd):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist_val = jnp.sum(jnp.abs(diff), axis=0)
    is_identical = (dist_val < 1e-5).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_singleton = (copy_count <= 1.0).astype(jnp.float32)

    n_obj = dist_val.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist_val) * (1.0 - eye)
    crowding = jnp.sum(sim, axis=1)

    return softmax_prior(
        w_singleton * is_singleton + w_features * ctx.feature_count - w_crowd * crowding
    )


def choice_probs(params, ctx):
    prior = compute_prior(
        ctx, params["w_singleton"], params["w_features"], params["w_crowd"]
    )
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
