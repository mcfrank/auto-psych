"""Resource-bounded pragmatic listener with uniform referent prior.

Refines bounded_capacity_listener by replacing the empirical visual salience prior
with an uninformative uniform prior over candidate referents. Listeners have bounded
cognitive resources for pragmatic reasoning, balancing communicative alignment with
the speaker against the mental cost of updating away from their default literal
interpretation, without imposing extraneous feature-count or familiarization weights.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.Gamma(2.0, 1.0),
    "capacity": dist.Beta(2.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def S1[u: UTT, r: OBJ](alpha, lex: ...):
    speaker: given(r in OBJ, wpp=1)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L_bounded[u: UTT, r: OBJ](alpha, capacity, lex: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=L0[u, r](lex)
        * exp(capacity * log(S1[u, r](alpha, lex) + {EPS})),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L_bounded(params["alpha"], params["capacity"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
