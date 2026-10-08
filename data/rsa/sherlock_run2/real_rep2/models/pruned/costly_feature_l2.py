"""RSA pragmatic listener at depth 2 with contextual feature extension costs.

Refines rsa_l2 by incorporating the utterance cost mechanism from
costly_feature_speaker into recursive pragmatic reasoning. Listeners reason at
depth 2 (inverting speaker S2 who simulates depth-1 pragmatic listener L1). At
both speaker levels (S1 and S2), utterances incur a production cost
proportional to their contextual extension—the fraction of objects in the
visual context that share that feature. Penalizing overextended features favors
distinctive, specific descriptions. On prior trials with no informative word,
choice is uniform. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost_weight": dist.Normal(0.0, 1.0),
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
            wpp=at(lex, u, r) * exp(alpha * (log(L0[u, r](lex) + {EPS}) - vec(cost, u))),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * (log(L1[u, r](alpha, lex, cost) + {EPS}) - vec(cost, u))),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    ext = jnp.sum(ctx.lex, axis=-1) / ctx.lex.shape[-1]
    cost = params["cost_weight"] * ext
    heard = L2(params["alpha"], ctx.lex, cost)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
