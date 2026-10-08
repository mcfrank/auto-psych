"""Population mixture of pragmatic and heuristic simplicity listeners.

Hypothesis: The participant population is cognitively heterogeneous, composed of a mixture
of pragmatic listeners and heuristic simplicity listeners. Pragmatic listeners infer the
intended referent by inverting an informative speaker through recursive mentalizing,
whereas heuristic listeners select among matching objects using a direct visual simplicity
bias that favors referents with fewer features. In the absence of an informative utterance,
choices reflect this population division as pragmatic listeners guess uniformly while
heuristic listeners select the simplest available object.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "simplicity": dist.LogNormal(0.0, 1.0),
    "pragmatic_share": dist.Beta(1.0, 1.0),
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
def L_heuristic[u: UTT, r: OBJ](lex: ..., weights: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(weights, r) + {EPS})
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    weights = softmax_prior(-params["simplicity"] * ctx.feature_count)
    uniform = jnp.full_like(weights, 1.0 / weights.shape[-1])

    pragmatic = L1(params["alpha"], ctx.lex)[ctx.utterance]
    heuristic = L_heuristic(ctx.lex, weights)[ctx.utterance]

    u_mix = params["pragmatic_share"] * pragmatic + (1.0 - params["pragmatic_share"]) * heuristic
    prior_mix = params["pragmatic_share"] * uniform + (1.0 - params["pragmatic_share"]) * weights

    choice = jnp.where(ctx.is_prior > 0, prior_mix, u_mix)
    return with_lapse(choice, params["lapse"])
