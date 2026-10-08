"""Pragmatic listener reasoning about an opportunity-cost speaker.

Speakers evaluate communicative descriptions in light of the opportunity cost of
foregone alternatives: producing an utterance incurs a cognitive penalty
proportional to the communicative informativeness of the best available
alternative description for that referent. A speaker who cannot name an object
any other way uses a shared feature without penalty, whereas a speaker
possessing an unambiguous alternative incurs an opportunity cost for choosing the
shared word. The pragmatic listener inverts this opportunity-cost-sensitive
speaker to infer the intended referent.
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


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, gamma, lex: ..., opp_cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - gamma * at(opp_cost, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    l0_prob = real_lex / (real_lex.sum(axis=1, keepdims=True) + EPS)
    n_utt = ctx.lex.shape[0]
    eye = jnp.eye(n_utt)
    masked = jnp.where(eye[:, :, None] > 0, 0.0, l0_prob[None, :, :])
    opp_cost = jnp.max(masked, axis=1) * (1.0 - ctx.is_sink[:, None])

    heard = L1(params["alpha"], params["gamma"], ctx.lex, opp_cost)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
