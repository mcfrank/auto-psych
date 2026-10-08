"""Contextual isolation listener: prior grounded in the Von Restorff isolation effect.

People evaluate candidate referents against a prior grounded in contextual
feature isolation rather than raw feature counts or partial contrast. In visual
reference displays, an object's communicative salience is governed by the Von
Restorff isolation effect: objects occupying isolated, distinctive positions in
feature space—measured by average pairwise feature distance to all competitors
across both shared attributes and contrasting omissions—capture attention and
prior expectation. Pragmatic listeners and speakers share this contextual
isolation prior as common ground, favoring isolated referents on prior displays
and updating counterfactually upon hearing referring expressions.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_isolation": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
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
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    non_sink_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    features = non_sink_lex.T
    n_obj = features.shape[0]
    diff = jnp.abs(features[:, None, :] - features[None, :, :])
    dist_matrix = jnp.sum(diff, axis=-1)
    denom = jnp.maximum(1.0, float(n_obj - 1))
    isolation = jnp.sum(dist_matrix, axis=-1) / denom

    prior = softmax_prior(
        params["w_isolation"] * isolation + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
