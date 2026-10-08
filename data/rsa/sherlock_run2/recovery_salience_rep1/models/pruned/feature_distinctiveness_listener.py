"""RSA pragmatic listener (depth 1) with a contextual feature-distinctiveness prior.

Listeners maintain an inductive prior over referents based on feature distinctiveness
relative to the surrounding scene: features that are unique or rare across the
context contribute high distinctiveness, while features shared across many objects
contribute low distinctiveness. In L1, the speaker draws referents according to
this contextual distinctiveness prior, and on uninformative prior trials the listener
relies directly on this prior.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "distinctiveness": dist.Normal(0.0, 1.0),
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
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    lex_no_sink = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    extension = jnp.sum(lex_no_sink, axis=1, keepdims=True)
    extension_safe = jnp.where(extension > 0, extension, 1.0)
    distinctiveness = jnp.sum(lex_no_sink / extension_safe, axis=0)
    prior = softmax_prior(params["distinctiveness"] * distinctiveness)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
