"""RSA pragmatic listener at depth 2 with base-rate prior and speaker ambiguity cost.

Refinement of base_rate_prior_l2: extends the model with an ambiguity cost
on the speaker (from ambiguity_cost_speaker). Speakers penalize utterances
in proportion to how many competing referents they describe, and depth-2
listeners invert this cost-sensitive speaker while integrating empirical
familiarization base rates.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost": dist.LogNormal(-1.0, 1.0),
    "base_rate_weight": dist.Normal(0.0, 2.0),
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


@memo
def L2[u: UTT, r: OBJ](alpha, costs: ..., lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, costs, lex, prior) + {EPS})
                - vec(costs, u)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    extension = jnp.sum(ctx.lex * (1.0 - ctx.is_sink)[:, None], axis=1)
    costs = params["cost"] * jnp.maximum(0.0, extension - 1.0)
    prior = softmax_prior(params["base_rate_weight"] * ctx.familiarization)
    heard = L2(params["alpha"], costs, ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
