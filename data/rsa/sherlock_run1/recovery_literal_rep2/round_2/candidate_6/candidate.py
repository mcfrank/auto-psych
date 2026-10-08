"""Valence-modulated salience listener with softmax decision rationality.

Refines valence_salience_listener by adding softmax decision rationality to the
pragmatic listener: rather than strictly probability-matching posterior beliefs
about the speaker's intended referent, listeners apply a softmax choice policy
governed by an independent decision rationality parameter beta. This separates
the speaker's communicative rationality from the listener's choice noise while
preserving the evaluative stance framing where speaker valence modulates prior
expectations over referents.

Differences from valence_salience_listener:
- Refined model: valence_salience_listener
- Recursion depth: depth 1 (choice_probs calls L1, unchanged).
- Parameters added: beta ~ LogNormal(0.0, 1.0) governing listener decision rationality.
- Parameters removed: none.
- Listener choice in L1: listener chooses r in OBJ with probability proportional to
  exp(beta * log(Pr[speaker.r == r] + EPS)) instead of matching Pr[speaker.r == r] directly.
- All other memo agents, distributions, and prior terms are unchanged.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    feat_weight = params["w_features"] + params["w_valence"] * ctx.valence
    prior = softmax_prior(
        feat_weight * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], params["beta"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
