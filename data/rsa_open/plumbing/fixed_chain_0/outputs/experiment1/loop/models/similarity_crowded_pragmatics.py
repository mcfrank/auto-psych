"""Shepard-Tversky similarity-crowded pragmatic listener and speaker.

Candidate referents with similar feature vectors and candidate words with similar
scene extensions crowd each other according to an exponential similarity kernel
exp(-gamma * d). Inside the communicative hierarchy, listeners divide reference
weight across each object's perceptual similarity cluster (1 / obj_crowding) and
speakers divide lexical choice weight across each word's extension similarity
cluster among words true of the target (1 / utt_crowding). At the top level and
on wordless prior trials, listeners favor perceptually uncrowded pop-out referents
proportional to obj_crowding ** (-beta).
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.5, 0.5),
    "gamma": dist.LogNormal(0.0, 0.75),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., inv_obj_crowd: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(inv_obj_crowd, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., lex_unif: ..., inv_obj_crowd: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(inv_obj_crowd, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex_unif, u, r)
            * exp(alpha * log(L0[u, r](lex, inv_obj_crowd) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., lex_unif: ..., inv_obj_crowd: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, lex_unif, inv_obj_crowd) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    gamma = params["gamma"]
    real_mask = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_mask[:, None]

    obj_diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    obj_dist = jnp.sum(obj_diff, axis=0)
    obj_sim = jnp.exp(-gamma * obj_dist)
    obj_crowding = jnp.sum(obj_sim, axis=1)
    inv_obj_crowd = 1.0 / obj_crowding
    prior = softmax_prior(-params["beta"] * jnp.log(obj_crowding))

    utt_diff = jnp.abs(real_lex[:, None, :] - real_lex[None, :, :])
    utt_dist = jnp.sum(utt_diff, axis=2)
    utt_sim = jnp.exp(-gamma * utt_dist) * (real_mask[:, None] * real_mask[None, :])
    utt_crowding = jnp.maximum(utt_sim @ real_lex, 1.0)
    lex_unif = ctx.lex / utt_crowding

    heard = L2(params["alpha"], ctx.lex, lex_unif, inv_obj_crowd, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
