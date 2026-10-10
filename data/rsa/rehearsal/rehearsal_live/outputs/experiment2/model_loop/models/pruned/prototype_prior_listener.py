"""Prototype prior pragmatic listener.

Listeners interpret referring expressions by modeling a communicative speaker
whose prior choice of an intended referent is driven by contextual typicality
(prototype theory). Candidate referents are evaluated by their similarity to the
ensemble prototype (the central tendency of visual features across display
competitors), with typical exemplars serving as accessible communicative
defaults. Pragmatic listeners invert this typicality-guided speaker via Bayes'
rule, combining the contextual prototype prior with linguistic informativeness.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_proto": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_prototype_similarity(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    feats = jnp.vstack([real_lex, (1.0 - ctx.grayscale)[None, :], ctx.familiarization[None, :]])
    centroid = jnp.mean(feats, axis=1, keepdims=True)
    diff = feats - centroid
    dist_sq = jnp.sum(diff ** 2, axis=0)
    return -dist_sq


def choice_probs(params, ctx):
    proto_sim = compute_prototype_similarity(ctx)
    prior = softmax_prior(params["w_proto"] * proto_sim)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
