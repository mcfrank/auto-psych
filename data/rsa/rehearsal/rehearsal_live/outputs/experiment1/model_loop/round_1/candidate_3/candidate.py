"""Pragmatic listener reasoning about an opportunity-cost-averse speaker.

Speakers in reference games evaluate candidate referring expressions relative to
the best available alternative description for the target referent, incurring an
opportunity cost whenever they produce a suboptimal, ambiguous word when a more
informative distinguishing feature was available. Pragmatic listeners invert this
opportunity-cost-averse speaker, inferring that when a speaker produces an ambiguous
expression, they lacked a more informative alternative for that referent. On
uninformative trials where no informative word is given, listeners expect speakers
to target referents with higher baseline communicative nameability.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_regret": dist.Normal(0.0, 2.0),
    "w_prior": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_regret, lex: ..., regret: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - w_regret * at(regret, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_regret_and_nameability(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    extension = jnp.sum(real_lex, axis=1, keepdims=True)
    real_L0 = real_lex / jnp.maximum(extension, 1.0)
    best_L0 = jnp.max(real_L0 * real_lex, axis=0, keepdims=True)
    regret = (best_L0 - real_L0) * real_lex
    nameability = best_L0[0, :]
    return regret, nameability


def choice_probs(params, ctx):
    regret, nameability = compute_regret_and_nameability(ctx)
    prior = softmax_prior(params["w_prior"] * nameability)
    heard = L1(params["alpha"], params["w_regret"], ctx.lex, regret)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
