"""Pragmatic listener at depth 2 with positive-feature gated topic salience and crowded-utterance prior retention."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_salience": dist.LogNormal(0.0, 0.5),
}


@memo
def L0[u: UTT, r: OBJ](avail: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(avail, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, avail: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(avail, u, r) * exp(alpha * log(L0[u, r](avail) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, avail: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(avail, u, r) * exp(alpha * log(L1[u, r](alpha, avail, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask

    # Unnormalized summed exponential dissimilarity (on a fixed 2-competitor reference scale),
    # gated by positive feature presence so featureless objects do not falsely pop out.
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    pos_gate = 1.0 - 0.25 * jnp.exp(-ctx.feature_count)
    contrast = 0.5 * pos_gate * jnp.sum(1.0 - jnp.exp(-pair_dist), axis=1)

    salience = contrast + 0.22 * ctx.feature_count
    prior = softmax_prior(params["w_salience"] * salience)

    # Feature-proportional lexical competition during speaker retrieval
    fc_excess = jnp.maximum(ctx.feature_count - 1.0, 0.0)[None, :]
    true_avail = 1.0 / (1.0 + 0.5 * fc_excess)
    avail = ctx.is_sink[:, None] * ctx.lex + real_mask * (ctx.lex * true_avail)

    prag = L2(params["alpha"], avail, prior)[ctx.utterance]

    # When the heard word is over-extended across >2 objects, the listener retains a small
    # fraction of the pre-utterance topic prior over the scene.
    heard_ext = jnp.sum(real_lex[ctx.utterance])
    retain = 0.0055 * jnp.maximum(heard_ext - 2.0, 0.0)
    heard = (1.0 - retain) * prag + retain * prior

    return jnp.where(ctx.is_prior > 0, prior, heard)
