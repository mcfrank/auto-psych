"""Pragmatic listener reasoning about communication with graded semantic truth values.

In natural human communication, words do not possess sharp binary truth values that apply
identically to all objects possessing a feature; instead, truth values are graded according
to semantic specificity. A descriptive word achieves maximal truth for an object that it
describes exhaustively, but its degree of truth degrades as candidate referents possess
additional unmentioned features. The foundational literal listener interprets words according
to this graded semantic fit, a communicative speaker anticipates this graded-truth listener,
and the pragmatic listener inverts the speaker to resolve referential ambiguity.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](coverage: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(coverage, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., coverage: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](coverage) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    extra = jnp.maximum(0.0, ctx.feature_count - 1.0)
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None]) * jnp.exp(-extra[None, :])
    sink_lex = ctx.lex * ctx.is_sink[:, None]
    coverage = real_lex + sink_lex

    heard = L1(params["alpha"], ctx.lex, coverage)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
