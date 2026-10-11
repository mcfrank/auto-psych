"""Pragmatic listener reasoning with a communicative expressibility referent prior.

Listeners expect a speaker in a reference game to intend referents with higher communicative
expressibility—objects that can be successfully and discriminatively named by the available
context vocabulary. The prior over referents is determined by the cumulative precision of
each object's distinguishing features, favoring uniquely expressible objects over featureless
or highly ambiguous distractors. Pragmatic listeners invert a rational speaker who chooses true
utterances with respect to this expressibility-weighted referent prior.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_express": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_expressibility(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Compute the cumulative communicative expressibility of each referent in context.

    Each real word's precision is inversely proportional to its contextual extension.
    An object's expressibility is the sum of precisions of the words that describe it.
    """
    real_words = 1.0 - is_sink
    real_lex = lex * real_words[:, None]
    ext = jnp.sum(real_lex, axis=1)
    prec = jnp.where(ext > 0, 1.0 / jnp.maximum(ext, 1.0), 0.0) * real_words
    return jnp.sum(real_lex * prec[:, None], axis=0)


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


def choice_probs(params, ctx):
    obj_express = compute_expressibility(ctx.lex, ctx.is_sink)
    prior = softmax_prior(params["w_express"] * obj_express)
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
