"""Depth-2 pragmatic listener with contextual cue validity graded truth values.

People evaluate referring expressions using graded truth values rather than binary
Boolean semantics: an utterance's semantic applicability to an object diminishes when
that object possesses alternative features that are far more distinctive (narrower in
extension) within the display context. For each candidate referent, a feature's semantic
applicability decays exponentially with its excess extension relative to the object's
most diagnostic attribute. A depth-2 pragmatic listener inverts a speaker who communicates
using these diagnostic truth values. On uninformative prior trials, choice is uniform
guessing. A lapse parameter mixes in random choice.
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


def compute_diagnostic_lex(lex: jnp.ndarray, is_sink: jnp.ndarray, gamma) -> jnp.ndarray:
    """Compute graded truth values where applicability diminishes with excess extension."""
    real_feat = lex * (1.0 - is_sink)[:, None]
    ext = jnp.sum(real_feat, axis=1)
    masked_ext = jnp.where(real_feat > 0, ext[:, None], 1e5)
    min_ext = jnp.min(masked_ext, axis=0)
    excess = jnp.maximum(ext[:, None] - min_ext[None, :], 0.0)
    precision = jnp.exp(-gamma * excess)
    real_truth = lex * precision
    return jnp.where(is_sink[:, None] > 0, lex, real_truth)


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


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    diag_lex = compute_diagnostic_lex(ctx.lex, ctx.is_sink, params["gamma"])
    heard = L2(params["alpha"], diag_lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
