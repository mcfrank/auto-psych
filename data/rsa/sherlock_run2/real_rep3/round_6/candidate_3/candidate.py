"""Pragmatic listener reasoning over an evaluative visual elaboration prior.

Referential expectations are grounded in an evaluative appraisal of candidate
referents along a unified dimension of visual elaboration, where discrete
features and chromatic color contribute positive visual ornamentation. In neutral
communication, listeners assume a baseline positive preference for embellished
referents; positive framing ('favorite') amplifies this preference, while negative
framing ('least favorite') inverts it toward plain, feature-sparse, and desaturated
items. Pragmatic listeners invert a speaker whose referential expectations reflect
this evaluative utility at depth 2 of recursive reasoning.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_base": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 2.0),
    "w_color": dist.Normal(0.0, 2.0),
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
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_prior(ctx, w_base, w_valence, w_color):
    color_prominence = 1.0 - ctx.grayscale
    elaboration = ctx.feature_count + w_color * color_prominence
    eval_weight = w_base + w_valence * ctx.valence
    return softmax_prior(eval_weight * elaboration)


def choice_probs(params, ctx):
    prior = compute_prior(
        ctx,
        params["w_base"],
        params["w_valence"],
        params["w_color"],
    )
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
