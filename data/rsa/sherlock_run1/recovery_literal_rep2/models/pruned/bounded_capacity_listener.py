"""Resource-bounded pragmatic listener with information-theoretic processing capacity.

Under bounded cognitive resources, listeners deviate from their default literal
interpretation toward the speaker's communicative intent only to the extent
permitted by their processing capacity. Formulated as information-theoretically
bounded rationality, the listener balances communicative alignment with the
speaker against the cognitive cost of updating away from the literal baseline,
weighting the speaker's communicative likelihood by a cognitive capacity parameter.
When capacity is constrained (capacity < 1), choices remain anchored near the
literal interpretation.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.Gamma(2.0, 1.0),
    "capacity": dist.Beta(2.0, 2.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def S1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    speaker: given(r in OBJ, wpp=1)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L_bounded[u: UTT, r: OBJ](alpha, capacity, lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=L0[u, r](lex, prior)
        * exp(capacity * log(S1[u, r](alpha, lex, prior) + {EPS})),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    heard = L_bounded(params["alpha"], params["capacity"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
