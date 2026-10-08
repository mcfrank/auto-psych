"""Pragmatic listener at depth 2 with communicative describability topic prior.

Listeners assume speakers preferentially refer to objects that can be successfully and
unambiguously communicated. An object's prior describability is the maximum literal
informativeness (L0 posterior) among true non-sink features available for that object.
A depth-2 listener L2 inverts an informative speaker S2 simulating a pragmatic listener L1,
with speakers at both levels evaluating candidate referents using this describability prior.
On prior trials with no informative word, choice follows the describability prior. A lapse
parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_describable": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def object_describability(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Maximum literal informativeness among true features for each object."""
    features = lex * (1.0 - is_sink)[:, None]
    count = jnp.sum(lex, axis=1, keepdims=True)
    l0 = features / jnp.maximum(count, 1.0)
    return jnp.max(l0, axis=0)


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    desc = object_describability(ctx.lex, ctx.is_sink)
    prior = softmax_prior(params["w_describable"] * desc)
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
