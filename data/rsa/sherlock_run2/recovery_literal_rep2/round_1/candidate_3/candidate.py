"""RSA model with speaker utterance cost penalizing specific features.

Speakers balance informativeness against an utterance cost for producing
specific features that apply to only a single referent in the context (e.g.
greater retrieval effort for narrow descriptors vs shared features). The
pragmatic listener reasons about this cost-sensitive speaker, moderating
scalar implicatures based on the speaker's trade-off between informativeness
and specificity cost. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost_specific": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}) - vec(cost, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    extension = ctx.lex.sum(axis=1)
    cost = params["cost_specific"] * (extension == 1.0).astype(jnp.float32) * (1.0 - ctx.is_sink)
    heard = L1(params["alpha"], ctx.lex, cost)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
