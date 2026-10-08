"""Pragmatic listener with graded truth values based on feature specificity.

People evaluate referring expressions with graded rather than binary Boolean truth values:
a feature word's semantic applicability diminishes as the number of additional visual features
on an object increases. The literal listener L0 evaluates referents using these graded truth
values, favoring objects whose features are not diluted by extraneous attributes. A pragmatic
listener L1 inverts an informative speaker S1 who communicates using this graded semantics.
With no informative word, choice is uniform guessing. A lapse parameter mixes in uniform noise.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.Normal(0.0, 1.0),
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
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_graded_lex(lex: jnp.ndarray, feature_count: jnp.ndarray, is_sink: jnp.ndarray, gamma) -> jnp.ndarray:
    """Compute graded truth values where applicability diminishes with additional features."""
    fc = jnp.maximum(feature_count, 1.0)
    precision = jnp.exp(-gamma * (fc - 1.0))
    real_truth = lex * precision[None, :]
    return jnp.where(is_sink[:, None] > 0, lex, real_truth)


def choice_probs(params, ctx):
    graded_lex = compute_graded_lex(ctx.lex, ctx.feature_count, ctx.is_sink, params["gamma"])
    heard = L1(params["alpha"], graded_lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
