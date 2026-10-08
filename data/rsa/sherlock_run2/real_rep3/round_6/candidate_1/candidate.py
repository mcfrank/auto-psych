"""Cognitive hierarchy listener: listeners model a speaker with bounded reasoning depth.

Standard RSA assumes a fixed recursion depth (depth 1 or depth 2). Under cognitive
hierarchy theory, resource-limited listeners recognize that speakers vary in communicative
depth. The listener models the speaker as a mixture over reasoning depths: with probability
(1 - p_informative) the speaker is a truthful literal speaker (S0, naming any true feature of
the intended referent), and with probability p_informative the speaker is an informative
strategic speaker (S1, optimizing informativeness for a literal listener). The listener
inverts this bounded-depth speaker to resolve reference.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(2.0, 1.0),
    "p_informative": dist.Beta(1.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L_bounded[u: UTT, r: OBJ](s_matrix: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(s_matrix, u, r)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    # S0: truthful literal speaker naming any valid feature
    w0 = ctx.lex
    s0 = w0 / (jnp.sum(w0, axis=0, keepdims=True) + EPS)

    # S1: informative pragmatic speaker optimizing for L0
    l0 = L0(ctx.lex)
    w1 = ctx.lex * jnp.exp(params["alpha"] * jnp.log(l0 + EPS))
    s1 = w1 / (jnp.sum(w1, axis=0, keepdims=True) + EPS)

    # Bounded speaker: mixture between literal and informative communicative depths
    p = params["p_informative"]
    s_mix = (1.0 - p) * s0 + p * s1

    # Listener inverts this depth-bounded speaker
    heard = L_bounded(s_mix)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
