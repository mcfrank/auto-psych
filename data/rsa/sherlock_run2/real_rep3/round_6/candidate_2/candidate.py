"""Pragmatic listener reasoning under graded semantic truth values.

In standard RSA, words have sharp, all-or-nothing truth conditions. Under the
graded truth hypothesis, semantic truth values are continuous: words have full
semantic truth for minimal, prototypical exemplars that possess only the target
feature, but exhibit graded truth values that decay exponentially as objects
accumulate extraneous, unmentioned features. Both the literal listener and the
pragmatic speaker reason over this graded semantic landscape.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "decay": dist.HalfNormal(1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](graded_lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(graded_lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, graded_lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(graded_lex, u, r) * exp(alpha * log(L0[u, r](graded_lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    is_real = 1.0 - ctx.is_sink
    extra_features = jnp.maximum(0.0, ctx.feature_count - 1.0)
    graded_weight = jnp.exp(-params["decay"] * extra_features)
    graded_lex = jnp.where(
        is_real[:, None] > 0.5,
        ctx.lex * graded_weight[None, :],
        ctx.lex,
    )
    heard = L1(params["alpha"], graded_lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
