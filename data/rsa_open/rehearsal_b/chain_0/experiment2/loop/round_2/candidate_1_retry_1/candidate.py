"""Pragmatic listener with communicative topic discriminability prior.

Listeners interpret referring expressions by modeling what speakers are likely to
talk about before an utterance is heard: a referential prior over objects grounded
in communicative discriminability. Before hearing a word, listeners expect speakers
to choose topics that can be successfully distinguished in context, assigning prior
probability to each candidate referent according to the maximal literal communicative
recovery of its features. On uninformative displays or universal utterances, this topic
prior directs belief toward uniquely discriminable objects without relying on spurious
visual singleton heuristics that distort scalar implicatures.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_discriminability": dist.Normal(0.0, 1.0),
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
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_discriminability(ctx):
    real_mask = 1.0 - ctx.is_sink[:, None]
    real_lex = ctx.lex * real_mask
    ext = jnp.sum(real_lex, axis=1, keepdims=True)
    safe_ext = jnp.where(ext > 0, ext, 1.0)
    l0 = real_lex / safe_ext
    return jnp.max(l0, axis=0)


def listener(params, ctx):
    disc = compute_discriminability(ctx)
    prior = softmax_prior(params["w_discriminability"] * disc)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
