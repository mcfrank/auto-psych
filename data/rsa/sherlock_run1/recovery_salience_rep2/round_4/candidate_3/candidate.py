"""Noisy-channel pragmatic listener.

Listeners interpret referring expressions under perceptual transmission noise,
marginalizing over a speaker's intended utterance transmitted across a noisy channel.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "channel_fidelity": dist.LogNormal(1.5, 0.75),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, channel_fidelity, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u_int in UTT, wpp=at(lex, u_int, r) * exp(alpha * log(L0[u_int, r](lex) + {EPS}))),
        channel: knows(speaker.u_int),
        channel: chooses(u_rec in UTT, wpp=exp(channel_fidelity * (u_rec == speaker.u_int))),
    ]
    listener: observes [channel.u_rec] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L1(params["alpha"], params["channel_fidelity"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
