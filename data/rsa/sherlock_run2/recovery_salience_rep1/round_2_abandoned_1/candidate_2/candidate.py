"""Capacity-limited listener: pragmatic reasoning precision is diluted by counterfactual feature load.

Hypothesis: Listeners have finite cognitive capacity that is diluted by the
counterfactual complexity of the display. Evaluating candidate referents with
additional alternative features divides the listener's attentional resources,
attenuating the effective rationality of their pragmatic inference.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "load_cost": dist.LogNormal(0.0, 1.0),
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


def choice_probs(params, ctx):
    matching = ctx.lex[ctx.utterance]
    alt_features = jnp.maximum(0.0, ctx.feature_count - 1.0)
    load = jnp.sum(matching * alt_features)
    alpha_eff = params["alpha"] / (1.0 + params["load_cost"] * load)
    heard = L1(alpha_eff, ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    choice = jnp.where(ctx.is_prior > 0, uniform, heard)
    return with_lapse(choice, params["lapse"])
