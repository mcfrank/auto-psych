"""Pragmatic listener combining familiarization base rates, referent simplicity, and speaker ambiguity cost.

Refinement of base_rate_simplicity_listener: Communicators integrate empirical base
rates established during familiarization with an inductive feature-simplicity prior
over referents, while speakers also incur an explicit communicative cost when
producing ambiguous words that apply to multiple objects in the scene (taken from
costly_feature_speaker).
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost": dist.LogNormal(0.0, 1.0),
    "simplicity": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, cost, lex: ..., costs: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}) - cost * vec(costs, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    simplicity_prior = softmax_prior(params["simplicity"] * ctx.feature_count)
    uniform = jnp.full_like(ctx.familiarization, 1.0 / ctx.familiarization.shape[-1])
    raw_base_rate = jnp.where(ctx.has_familiarization > 0, ctx.familiarization, uniform)
    combined = raw_base_rate * simplicity_prior
    prior = (combined + EPS) / jnp.sum(combined + EPS)
    costs = (1.0 - ctx.is_sink) * jnp.maximum(0.0, jnp.sum(ctx.lex, axis=1) - 1.0)
    heard = L1(params["alpha"], params["cost"], ctx.lex, costs, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
