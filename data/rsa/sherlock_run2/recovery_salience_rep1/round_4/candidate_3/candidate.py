"""Pragmatic listener with noisy-channel communication.

Hypothesis: Listeners interpret referential statements under the expectation that
communicative transmission is subject to channel noise and speech production slips.
Rather than assuming the heard word was transmitted deterministically, the listener
models a speaker whose intended utterance passes through a noisy transmission channel,
jointly inferring the speaker's communicative intent and intended referent. In the
absence of an informative utterance, choices remain uniform across available objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "channel_noise": dist.Beta(1.0, 9.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., channel_matrix: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u_int in UTT, wpp=at(lex, u_int, r) * exp(alpha * log(L0[u_int, r](lex) + {EPS}))),
        channel: knows(speaker.u_int),
        channel: chooses(u_obs in UTT, wpp=at(channel_matrix, u_obs, speaker.u_int)),
    ]
    listener: observes [channel.u_obs] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    n_utt = ctx.lex.shape[0]
    eye = jnp.eye(n_utt)
    noise = params["channel_noise"]
    # Channel transmission: with probability 1 - noise faithful transmission,
    # and with probability noise distributed across alternative vocabulary words.
    channel_matrix = (1.0 - noise) * eye + (noise / n_utt) * jnp.ones((n_utt, n_utt))

    heard = L1(params["alpha"], ctx.lex, channel_matrix)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
