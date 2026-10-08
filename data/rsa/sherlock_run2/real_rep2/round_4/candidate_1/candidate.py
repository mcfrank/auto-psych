"""Pragmatic listener at depth 2 reasoning about communicative aspect agreement.

Rather than assuming communication is strictly about identifying object identity,
the listener models a speaker whose goal is to convey the referent's constituent
aspects (visual features). An utterance's communicative utility is evaluated by the
expected aspect agreement achieved under the listener's interpretation—rewarding
descriptions that successfully communicate the target's visual properties even when
referents share common features. Reasoning recurses to depth 2: a pragmatic listener L2
inverts an informative speaker S2 who anticipates how a depth-1 listener L1 interprets
utterances with respect to expected aspect agreement. On uninformative prior trials,
choice is uniform. A lapse parameter mixes in uniform guessing.
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
def L1[u: UTT, r: OBJ](alpha, lex: ..., aspect_utility: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(at(aspect_utility, u, r) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., aspect_utility: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(at(aspect_utility, u, r) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def aspect_agreement_matrix(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Aspect agreement between referent r and candidate referent r_prime."""
    features = lex * (1.0 - is_sink)[:, None]
    fc = jnp.maximum(jnp.sum(features, axis=0, keepdims=True), 1.0)
    shared = jnp.matmul(features.T, features)
    is_plain = (jnp.sum(features, axis=0) == 0).astype(jnp.float32)
    plain_agree = jnp.outer(is_plain, is_plain)
    return (shared / fc.T) + plain_agree


def choice_probs(params, ctx):
    A = aspect_agreement_matrix(ctx.lex, ctx.is_sink)
    count = jnp.sum(ctx.lex, axis=1, keepdims=True)
    l0 = ctx.lex / jnp.maximum(count, 1.0)
    u1 = jnp.matmul(l0, A.T)
    l1 = L1(params["alpha"], ctx.lex, u1)
    u2 = jnp.matmul(l1, A.T)
    heard = L2(params["alpha"], ctx.lex, u2)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
