"""Pragmatic listener reasoning under semantic feature attribution uncertainty.

Listeners in reference games maintain uncertainty about which visual feature a referring
expression picks out. Rather than assuming that an utterance deterministically maps to a
single feature dimension, people assign a baseline probability to the heard expression
picking out other context features present in the scene, weighting candidate referents
by both their matching features and their overall visual feature density. Pragmatic
speakers and listeners recursively reason over this feature attribution uncertainty when
producing and resolving referring expressions.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "theta": dist.Beta(1.0, 9.0),
    "w_feature": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](soft_lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(soft_lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, soft_lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(soft_lex, u, r)
            * exp(alpha * log(L0[u, r](soft_lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, soft_lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(soft_lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, soft_lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_soft_lex_and_prior(ctx, theta, w_feature):
    n_words = jnp.maximum(jnp.sum(1.0 - ctx.is_sink), 1.0)
    feat_density = ctx.feature_count / n_words
    soft_real = (1.0 - theta) * ctx.lex + theta * feat_density[None, :]
    soft_lex = jnp.where(ctx.is_sink[:, None] > 0, ctx.lex, soft_real)
    prior = softmax_prior(w_feature * ctx.feature_count)
    return soft_lex, prior


def choice_probs(params, ctx):
    soft_lex, prior = compute_soft_lex_and_prior(
        ctx, params["theta"], params["w_feature"]
    )
    heard = L2(params["alpha"], soft_lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
