"""Pragmatic listener inverting a regret-sensitive speaker.

Speakers evaluate candidate utterances by balancing communicative informativeness
against the opportunity cost (regret) relative to the most informative description
available for the target referent. When choosing among true words, speakers incur a
penalty proportional to how much less informative the chosen word is compared to the
target's best available alternative description. Pragmatic listeners invert this
regret-sensitive speaker to interpret referring expressions. With no informative
word, choice is uniform guessing. A lapse parameter mixes in random choice.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_regret": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_regret(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Compute opportunity cost (regret) of each utterance for each target object.

    For each target object r, we determine the maximum literal informativeness
    I*(r) among real (non-sink) true features. The regret of true utterance u
    for object r is max(0, I*(r) - L0(u, r)). Sink utterances and false words have zero regret.
    """
    real_lex = lex * (1.0 - is_sink)[:, None]
    counts = jnp.sum(lex, axis=1, keepdims=True)
    l0 = real_lex / jnp.maximum(counts, 1.0)
    best_l0 = jnp.max(l0, axis=0, keepdims=True)  # (1, N_OBJ)
    regret = jnp.maximum(0.0, best_l0 - l0) * real_lex
    return regret


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_regret, lex: ..., regret: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * (log(L0[u, r](lex) + {EPS}) - w_regret * at(regret, u, r))),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    regret = compute_regret(ctx.lex, ctx.is_sink)
    heard = L1(params["alpha"], params["w_regret"], ctx.lex, regret)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
