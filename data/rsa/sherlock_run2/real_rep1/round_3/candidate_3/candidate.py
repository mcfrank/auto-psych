r"""Pragmatic listener with a softmax decision rule over beliefs.

In standard RSA, listeners are assumed to choose referents via strict probability
matching directly from their posterior beliefs (Pr[choice == r] = Pr[speaker.r == r]).
Under the softmax decision listener hypothesis, belief formation and motor action
selection are distinct cognitive processes: the listener infers posterior beliefs
via simple communicative reasoning and selects a referent via a softmax decision
rule with decision precision beta (Pr[choice == r] \propto exp(beta * Pr[speaker.r == r])).
This exponentially sharpens choices toward the most probable referent while
assigning an exponentially suppressed baseline choice probability (exp(0) = 1)
to non-matching distractors, naturally modeling decision noise.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(1.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * Pr[speaker.r == r]))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L1(params["alpha"], params["beta"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
