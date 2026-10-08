"""RSA depth-2 pragmatic listener with empirical base-rate prior.

Refines rsa_l2 by replacing the uniform prior over objects with an empirical
base-rate prior derived from familiarization frequencies when available.
Listeners at depth 2 invert a speaker S2 who simulates a pragmatic listener L1;
both speaker levels represent the prior over referents via the familiarization
distribution scaled by a sensitivity parameter base_rate_weight. On prior trials
with no informative word, choices follow the base-rate prior directly.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "base_rate_weight": dist.HalfNormal(1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    n_obj = ctx.familiarization.shape[0]
    uniform = jnp.full_like(ctx.familiarization, 1.0 / n_obj)
    fam_logits = params["base_rate_weight"] * jnp.log(ctx.familiarization + EPS)
    fam_prior = softmax_prior(fam_logits)
    prior = jnp.where(ctx.has_familiarization > 0, fam_prior, uniform)

    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
