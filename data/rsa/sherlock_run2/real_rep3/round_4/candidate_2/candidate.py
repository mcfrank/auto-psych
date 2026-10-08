"""Population mixture of pragmatic and perceptual listeners.

The participant population comprises two distinct cognitive types: pragmatic
listeners who resolve referring expressions via depth-2 recursive Theory of Mind
(inverting a speaker who simulates a depth-1 listener), and perceptual listeners
who select contextually isolated singleton objects that lack identical duplicates
among truth-conditional referents. Population choices reflect this underlying
mixture distribution, with a parameter governing the proportion of pragmatic
versus perceptual listeners. On uninformative trials, pragmatic listeners choose
uniformly while perceptual listeners select the contextually unique singleton.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 2.0),
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
def L_perc[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


def compute_singleton_prior(ctx, w_singleton):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    is_identical = (dist < 1e-5).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_singleton = (copy_count <= 1.0).astype(jnp.float32)
    return softmax_prior(w_singleton * is_singleton)


def choice_probs(params, ctx):
    prior_singleton = compute_singleton_prior(ctx, params["w_singleton"])
    prag_heard = L2(params["alpha"], ctx.lex)[ctx.utterance]
    perc_heard = L_perc(ctx.lex, prior_singleton)[ctx.utterance]

    heard = params["p_pragmatic"] * prag_heard + (1.0 - params["p_pragmatic"]) * perc_heard

    prior_uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    prior_pop = params["p_pragmatic"] * prior_uniform + (1.0 - params["p_pragmatic"]) * prior_singleton

    return with_lapse(jnp.where(ctx.is_prior > 0, prior_pop, heard), params["lapse"])
