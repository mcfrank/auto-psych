"""Pragmatic listener at depth 2 reasoning about a speaker with expected competitor confusion costs.

Speakers choose utterances to maximize communicative informativeness for a simulated
pragmatic listener while penalizing utterances by the expected visual similarity among
objects in the utterance's extension and the target. Listeners invert this speaker at
depth 2 with a prior favoring perceptually isolated objects and relative feature extremity.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import BIG, EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_cost": dist.Normal(0.0, 1.0),
    "w_extremity": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS}) - at(cost, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, lex, prior, cost) + {EPS})
                - at(cost, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_cost_and_prior(ctx, w_cost, w_extremity):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    feats = jnp.vstack(
        [real_lex, ctx.grayscale[None, :], ctx.familiarization[None, :]]
    )
    diff = jnp.abs(feats[:, :, None] - feats[:, None, :])
    dist_mat = jnp.sum(diff, axis=0)
    n_obj = dist_mat.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist_mat) * (1.0 - eye)

    # Expected competitor confusion: expected similarity of a literal listener's draw from u
    # to the target r, plus the expected similarity among objects in the extension of u.
    ext_size = jnp.maximum(jnp.sum(real_lex, axis=1, keepdims=True), 1.0)
    target_conf = (real_lex @ sim) / ext_size
    ext_conf = jnp.sum((real_lex @ sim) * real_lex, axis=1, keepdims=True) / ext_size
    cost = w_cost * (target_conf + ext_conf)

    # Perceptual isolation (minimum distance to any other object) and relative feature extremity
    isolation = jnp.min(dist_mat + eye * BIG, axis=1)
    fc = ctx.feature_count
    mean_fc = jnp.mean(fc)
    extremity = (fc - mean_fc) * (fc - mean_fc) / (fc + 1.0)

    prior = softmax_prior(w_cost * isolation + w_extremity * extremity)
    return cost, prior


def listener(params, ctx):
    cost, prior = compute_cost_and_prior(
        ctx, params["w_cost"], params["w_extremity"]
    )
    heard = L2(params["alpha"], ctx.lex, prior, cost)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
