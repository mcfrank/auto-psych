"""RSA pragmatic listener (depth 1) reasoning about a cost-sensitive speaker.

Speakers evaluate candidate utterances by balancing literal informativeness
against production cost, incurring a cost that scales with the feature's
extension across objects in the visual context. Pragmatic listeners invert this
cost-sensitive speaker, allowing utterance-level production effort to shape
referent choice alongside informativeness. With no informative word the choice
is uniform. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo
from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(1.0, 1.0),
    "cost_ambiguity": dist.Normal(0.0, 1.0),
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
    n_obj_true = jnp.sum(ctx.lex * (1.0 - ctx.is_sink)[:, None], axis=-1)
    ambiguity = jnp.maximum(0.0, n_obj_true - 1.0)
    cost = params["cost_ambiguity"] * ambiguity
    heard = L1(params["alpha"], ctx.lex, cost)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])