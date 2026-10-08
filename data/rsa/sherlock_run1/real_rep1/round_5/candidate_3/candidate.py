"""RSA pragmatic listener with a contextual contrast (von Restorff isolation) salience prior.

Listeners interpret referential descriptions by maintaining a prior expectation
that speakers refer to perceptually isolated objects that contrast with the visual context,
grounded in the von Restorff isolation effect. Before hearing an informative utterance,
listeners evaluate each candidate referent by its contextual contrast—the average
dissimilarity (feature distance) to all other objects in the display—prioritizing objects
that stand apart in feature space whether by possessing unique features, lacking common
features, or contrasting with duplicate competitors. Pragmatic listeners invert an
informative speaker who shares this contextual contrast prior across recursive reasoning depths.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_contrast": dist.Normal(0.0, 1.0),
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
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    feat = real_lex.T
    diff = (feat[:, None, :] - feat[None, :, :]) ** 2
    n_obj = ctx.lex.shape[1]
    contrast = jnp.sum(diff, axis=-1).sum(axis=1) / jnp.maximum(n_obj - 1.0, 1.0)

    prior = softmax_prior(
        params["w_contrast"] * contrast + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
