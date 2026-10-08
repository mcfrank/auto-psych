"""Distinctiveness heuristic listener with neutral prior: contextual distinctiveness heuristic without prior bias.

Listeners interpret referring expressions using a non-Bayesian perceptual distinctiveness heuristic
rather than recursive speaker simulation. When an utterance is heard, listeners restrict attention
to matching referents and select the object that is most contextually distinctive in the visual display,
possessing features that are rarest across the other objects. On prior trials without an informative
description, choice is guided by a neutral object prior that reflects familiarization base rates
without an intrinsic distinctiveness bias. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heuristic[u: UTT, r: OBJ](beta, lex: ..., distinct: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(beta * vec(distinct, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    features = (1.0 - ctx.is_sink)[:, None] * ctx.lex
    freq = features.sum(axis=1)
    safe_freq = jnp.where(freq > 0.0, freq, 1.0)
    distinctiveness = (features / safe_freq[:, None]).sum(axis=0)

    prior = softmax_prior(params["w_familiar"] * ctx.familiarization)
    heard = L_heuristic(params["beta"], ctx.lex, distinctiveness)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
