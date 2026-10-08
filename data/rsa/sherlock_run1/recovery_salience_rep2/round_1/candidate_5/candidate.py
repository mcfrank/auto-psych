"""RSA pragmatic listener at depth 2 with familiarization base-rate prior.

Refines rsa_l2 by incorporating familiarization base rates into the depth-2
listener's prior over referents. When objects have prior exposure frequencies,
the listener weights them by base_rate_weight against a uniform baseline and
uses this prior when inverting the depth-2 speaker, as well as choosing by it
on uninformative prior trials.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
    "base_rate_weight": dist.Beta(2.0, 2.0),
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


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    uniform = jnp.full_like(ctx.feature_count, 1.0 / ctx.feature_count.shape[0])
    fam_prior = (1.0 - params["base_rate_weight"]) * uniform + params["base_rate_weight"] * ctx.familiarization
    prior = jnp.where(ctx.has_familiarization > 0, fam_prior, uniform)
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
