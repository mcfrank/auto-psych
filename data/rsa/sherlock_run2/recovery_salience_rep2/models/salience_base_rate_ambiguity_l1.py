"""RSA pragmatic listener at depth 1 with inductive base-rate prior, feature complexity, and speaker ambiguity cost.

Refinement of salience_base_rate_ambiguity_l2: reduces recursion depth from depth 2 to depth 1.
Listeners interpret referential expressions by inverting a speaker who reasons about a literal
listener with cognitive ambiguity costs, while evaluating candidate referents through an inductive
prior combining empirical familiarization base rates and visual feature complexity.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost": dist.LogNormal(-1.0, 1.0),
    "base_rate_weight": dist.Normal(0.0, 2.0),
    "salience": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, costs: ..., lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS}) - vec(costs, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    extension = jnp.sum(ctx.lex * (1.0 - ctx.is_sink)[:, None], axis=1)
    costs = params["cost"] * jnp.maximum(0.0, extension - 1.0)
    prior = softmax_prior(
        params["base_rate_weight"] * ctx.familiarization
        + params["salience"] * ctx.feature_count
    )
    heard = L1(params["alpha"], costs, ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
