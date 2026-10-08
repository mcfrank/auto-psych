"""Pragmatic listener with feature surprisal salience prior.

Listeners evaluate candidate referents according to the contextual
informativeness (Shannon surprisal) of their visual features: contextually
rare or unique features carry high information content, whereas ubiquitous
features carry low information. Pragmatic listeners ground reference
resolution in this feature-surprisal prior, expecting speakers to refer to
high-information referents on uninformative trials and when resolving ambiguous
referring expressions.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_surprisal": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS}))),
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


def compute_surprisal_prior(ctx, w_surprisal):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_obj = ctx.lex.shape[1]
    counts = jnp.sum(real_lex, axis=1, keepdims=True)
    freq = counts / n_obj
    surprisal = -jnp.log(jnp.maximum(freq, 1e-6)) * real_words
    obj_surprisal = jnp.sum(real_lex * surprisal, axis=0)
    return softmax_prior(w_surprisal * obj_surprisal)


def choice_probs(params, ctx):
    prior = compute_surprisal_prior(ctx, params["w_surprisal"])
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
