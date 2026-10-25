"""Pragmatic listener reasoning with contextual visual surprisal.

Listeners interpret referring expressions by modeling what the speaker is likely to
talk about before an utterance is heard, forming an object prior grounded in
contextual visual surprisal. Candidate referents whose visual feature bundles
deviate most from the display's mean feature prevalence capture spontaneous
attention and serve as natural communicative topics. Communicative speakers choose
expressions by anticipating this visual prior, and listeners infer the intended
target through Bayesian inversion.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_surprisal": dist.Normal(0.0, 1.0),
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


def compute_surprisal(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    mean_f = jnp.mean(real_lex, axis=1, keepdims=True)
    diff = jnp.abs(real_lex - mean_f)
    return jnp.sum(diff * real_words, axis=0)


def listener(params, ctx):
    surprisal = compute_surprisal(ctx)
    prior = softmax_prior(params["w_surprisal"] * surprisal)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
