"""Pragmatic listener with a softmax decision rule over posterior beliefs.

Hypothesis: Listeners evaluate referential statements using pragmatic Bayesian
reasoning to form beliefs about intended referents, but convert those beliefs
into choices via a softmax decision rule rather than pure probability matching.
Governed by a decision precision parameter, this response rule allows listeners
to overmatch their posterior beliefs by sharpening preference toward the most
probable referent, or undermatch toward indifference under uncertainty. In the
absence of an informative utterance, choices remain uniform across available
objects.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "decision_precision": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L1(params["alpha"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    beliefs = jnp.where(ctx.is_prior > 0, uniform, heard)

    # Softmax decision rule over posterior beliefs
    log_beliefs = jnp.log(beliefs + EPS)
    decision_probs = jax.nn.softmax(params["decision_precision"] * log_beliefs)

    return with_lapse(decision_probs, params["lapse"])
