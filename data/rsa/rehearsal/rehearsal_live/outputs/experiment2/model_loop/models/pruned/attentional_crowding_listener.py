"""Pragmatic listener with attentional crowding constraints.

Listeners have capacity-limited visual attention that suffers from visual
crowding when candidate referents share features with similar display
competitors. Attentional capacity allocated to each referent decays with its
visual crowd size, so that visually isolated singletons receive focused
attentional gain while redundant items suffer attentional suppression.
Pragmatic listeners invert a communicative speaker who anticipates this
attentional crowding constraint, while choices on uninformative prior trials
directly track uncrowded visual attention.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.Normal(0.0, 1.0),
    "w_prior": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](gamma, lex: ..., crowd: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-gamma * vec(crowd, r)),
    )
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, gamma, lex: ..., crowd: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=exp(-gamma * vec(crowd, r))),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](gamma, lex, crowd) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_crowd_size(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    feats = jnp.vstack([real_lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = jnp.abs(feats[:, :, None] - feats[:, None, :])
    dist = jnp.sum(diff, axis=0)
    # Exponential similarity in feature space (Shepard's law)
    sim = jnp.exp(-dist)
    crowd_size = jnp.sum(sim, axis=1)
    return jnp.log(crowd_size)


def choice_probs(params, ctx):
    crowd = compute_crowd_size(ctx)
    prior = softmax_prior(-params["w_prior"] * crowd)
    heard = L1(params["alpha"], params["gamma"], ctx.lex, crowd)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard),
        params["lapse"],
    )
