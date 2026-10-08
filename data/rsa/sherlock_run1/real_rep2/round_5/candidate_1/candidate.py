
"""Shepard confusability speaker: psychological similarity modulates competitor interference.

People interpret referring expressions by modeling the speaker as evaluating
referent discriminability under psychological similarity, where competing objects
interfere not uniformly, but in proportion to their feature overlap with the
intended referent according to Shepard's law of generalization. On displays where
an uttered word applies to both a distinct singleton object and multiple duplicate
competitor objects, the current best model predicts listeners will counterfactually
avoid the singleton because the speaker withheld its alternative unique label,
whereas similarity-discounted confusability predicts listeners will strongly favor
the singleton because the duplicate competitors suffer severe mutual interference
while the singleton stands out as perceptually unconfusable.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 1.0),
    "delta": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, gamma, lex: ..., prior: ..., util: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * at(util, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(gamma * Pr[speaker.r == r]))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    features = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    n_features = jnp.maximum(1.0, jnp.sum(1.0 - ctx.is_sink))
    diff = jnp.abs(features[:, :, None] - features[:, None, :])
    dist_matrix = jnp.sum(diff, axis=0) / n_features
    sim = jnp.exp(-params["delta"] * dist_matrix)

    denom = jnp.sum(ctx.lex[:, None, :] * prior[None, None, :] * sim[None, :, :], axis=-1)
    target_num = ctx.lex * prior[None, :]
    l_shepard = target_num / (denom + EPS)
    util = jnp.log(l_shepard + EPS)

    heard = L1(params["alpha"], params["gamma"], ctx.lex, prior, util)[ctx.utterance]
    prior_choice = jax.nn.softmax(params["gamma"] * prior)
    return with_lapse(jnp.where(ctx.is_prior > 0, prior_choice, heard), params["lapse"])
