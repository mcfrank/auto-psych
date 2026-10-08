"""Pragmatic listener reasoning about a speaker who can choose to remain silent.

The speaker has the option of remaining silent (the sink utterance) rather than
speaking, incurring a baseline cost of silence. For an intended referent, the
speaker weighs the informativeness of true feature words against remaining silent.
The pragmatic listener inverts this speaker, taking the decision to speak into account.
On prior trials without an informative word, the listener chooses uniformly.
A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost_silence": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, cost_silence, lex: ..., is_sink: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=(1.0 - vec(is_sink, u)) * at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))
            + vec(is_sink, u) * exp(-cost_silence),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L1(params["alpha"], params["cost_silence"], ctx.lex, ctx.is_sink)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
