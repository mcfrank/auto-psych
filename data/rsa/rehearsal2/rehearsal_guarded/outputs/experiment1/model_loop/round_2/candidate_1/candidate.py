"""RSA pragmatic listener at depth 2 with visual feature surprisal salience.

Listeners compute an attentional prior over candidate referents grounded in
Shannon feature surprisal: features that are ubiquitous in the scene convey zero
information, while rare or unique features carry high surprisal (-log p(f)),
making objects that possess them visually salient. Pragmatic listeners invert
a speaker who shares this common-ground feature surprisal prior at depth 2.
Literal listener L0 operates on pure truth-conditional semantics.
On uninformative prior trials with no distinguishing word, choices follow
the feature surprisal prior directly. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_surprisal": dist.Normal(0.0, 1.0),
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
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_surprisal_prior(ctx, w_surprisal):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_obj = ctx.lex.shape[1]
    n_r = jnp.sum(real_lex, axis=1)
    p_u = jnp.where(n_r > 0, n_r / n_obj, 1.0)
    surprisal = jnp.where(n_r > 0, -jnp.log(p_u), 0.0)
    obj_surprisal = jnp.sum(real_lex * surprisal[:, None], axis=0)
    return softmax_prior(w_surprisal * obj_surprisal)


def choice_probs(params, ctx):
    prior = compute_surprisal_prior(ctx, params["w_surprisal"])
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
