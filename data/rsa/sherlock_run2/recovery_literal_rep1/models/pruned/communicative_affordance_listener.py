"""RSA pragmatic listener (depth 1) with communicative affordance topic prior.

Listeners model the speaker as choosing an intended referent based on communicative
affordance: before hearing an informative word, listeners assign higher prior probability
to referents that possess greater communicative expressibility, reasoning that a speaker
constrained to a single word prefers to discuss topics that can be communicated with high
discriminative precision. In pragmatic reference games, listeners coordinate with the
speaker on this topic prior, expecting the speaker's referential goal to favor objects
with distinctive communicative descriptors over ambiguous ones.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_topic": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
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
    mask = 1.0 - ctx.is_sink
    ext = jnp.sum(ctx.lex * mask[:, None], axis=1)
    acc = (ctx.lex * mask[:, None]) / jnp.maximum(1.0, ext[:, None])
    express = jnp.sum(acc, axis=0)
    prior = softmax_prior(
        params["w_topic"] * express + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
