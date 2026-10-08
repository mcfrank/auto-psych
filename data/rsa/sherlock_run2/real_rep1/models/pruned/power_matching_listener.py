r"""Pragmatic listener with power-law probability matching (generalized matching law).

In standard RSA, listeners are assumed to choose referents via strict probability
matching directly from their posterior beliefs (Pr[choice == r] = Pr[speaker.r == r]).
Under the power-law matching listener hypothesis, belief formation and decision
selection are distinct cognitive stages: the listener infers posterior beliefs
via communicative reasoning and selects a referent via a power-law matching rule
with response sensitivity gamma (Pr[choice == r] \propto Pr[speaker.r == r] ** gamma).
For gamma > 1 (overmatching), this sharpens choices toward the most probable referent
while strictly respecting semantic truth by assigning zero weight to impossible
referents, unlike an exponential softmax.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, gamma, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(
        r in OBJ,
        wpp=exp(gamma * log(Pr[speaker.r == r] + {EPS})),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L1(params["alpha"], params["gamma"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
