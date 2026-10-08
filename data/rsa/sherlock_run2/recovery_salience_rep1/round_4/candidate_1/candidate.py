"""Aspect goal listener: speaker communicates an aspect of the object rather than its identity.

Hypothesis: Speakers communicate about a specific aspect or feature of an object
rather than attempting to uniquely identify the object itself, prioritizing aspects
that are more distinctive in the visual scene. Listeners interpret utterances by
jointly reasoning about the speaker's communicative goal and intended referent,
inferring that mentioning a shared feature implies the speaker lacked any more
distinctive aspect to convey. In the absence of an informative message, listeners
choose uniformly among the available objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "goal_weight": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def aspect_listener[u: UTT, r: OBJ](alpha, lex: ..., aspect_prob: ..., aspect_weight: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(g in UTT, wpp=at(lex, g, r) * vec(aspect_weight, g)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * (exp(alpha * log(at(aspect_prob, u, g) + {EPS})) + 0.01)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    l0_weights = ctx.lex / jnp.maximum(EPS, jnp.sum(ctx.lex, axis=1, keepdims=True))
    aspect_prob = jnp.matmul(l0_weights, ctx.lex.T)

    extension = jnp.sum(ctx.lex, axis=1)
    n_obj = ctx.lex.shape[1]
    distinctiveness = jnp.log(n_obj / jnp.maximum(1.0, extension))
    aspect_weight = jnp.exp(params["goal_weight"] * distinctiveness)

    heard = aspect_listener(params["alpha"], ctx.lex, aspect_prob, aspect_weight)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
