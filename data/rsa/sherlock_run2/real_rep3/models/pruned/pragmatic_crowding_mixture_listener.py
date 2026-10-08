"""Population mixture of pragmatic and crowding-averse perceptual listeners.

The participant population comprises two distinct cognitive types: pragmatic
listeners who resolve referring expressions via depth-2 recursive Theory of Mind
(inverting a speaker who simulates a depth-1 pragmatic listener), and perceptual
listeners who select referents that escape visual attentional crowding from shared
feature overlap with distractors in the display. Population choices reflect this
underlying mixture distribution, with a parameter governing the proportion of
pragmatic versus crowding-averse listeners. On uninformative trials, pragmatic
listeners choose uniformly while perceptual listeners select referents that minimize
visual crowding.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.5, 0.5),
    "w_crowd": dist.HalfNormal(3.0),
    "p_pragmatic": dist.Beta(2.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L_crowd[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


def compute_crowding_prior(ctx, w_crowd):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)
    crowding = jnp.sum(sim, axis=1)
    return softmax_prior(-w_crowd * crowding)


def choice_probs(params, ctx):
    prior_crowd = compute_crowding_prior(ctx, params["w_crowd"])
    prag_heard = L2(params["alpha"], ctx.lex)[ctx.utterance]
    crowd_heard = L_crowd(ctx.lex, prior_crowd)[ctx.utterance]

    heard = params["p_pragmatic"] * prag_heard + (1.0 - params["p_pragmatic"]) * crowd_heard

    prior_uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    prior_pop = params["p_pragmatic"] * prior_uniform + (1.0 - params["p_pragmatic"]) * prior_crowd

    return with_lapse(jnp.where(ctx.is_prior > 0, prior_pop, heard), params["lapse"])
