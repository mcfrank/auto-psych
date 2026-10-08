"""Base-rate tracking listener (depth 1).

Listeners invert a softmax-rational speaker using empirical base rates
established during familiarization as a prior over referents. When no
informative utterance is heard (prior trials), choices track the base rates
directly; with an informative word, the base-rate prior weights the speaker's
communicative likelihood.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
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


def choice_probs(params, ctx):
    uniform = jnp.full_like(ctx.familiarization, 1.0 / ctx.familiarization.shape[-1])
    raw_prior = jnp.where(ctx.has_familiarization > 0, ctx.familiarization, uniform)
    prior = (raw_prior + EPS) / jnp.sum(raw_prior + EPS)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
