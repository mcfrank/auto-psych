"""Pragmatic listener with contrastive perceptual salience (outlier / isolation effect).

Hypothesis: Listeners evaluate candidate referents through the lens of contrastive
perceptual salience, expecting communicators to target objects that stand out as
distinctive contrasts or outliers against the visual context. An object's contrastive
salience increases with its average feature dissimilarity from the other objects
in the display, making isolated or odd-man-out objects more salient default referents
than objects clustered with visually similar competitors. In the absence of an
informative utterance, listeners choose directly according to this contrastive salience prior.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "contrast": dist.Normal(0.0, 1.0),
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
    # Features excluding sink utterance
    features = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    # Pairwise Manhattan distance between objects across features
    diff = jnp.abs(features.T[:, None, :] - features.T[None, :, :])
    dist_matrix = jnp.sum(diff, axis=-1)
    n_obj = ctx.lex.shape[1]
    # Average dissimilarity to all other objects in the scene
    contrast = jnp.sum(dist_matrix, axis=-1) / jnp.maximum(1.0, n_obj - 1.0)

    prior = softmax_prior(params["contrast"] * contrast)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    probs = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(probs, params["lapse"])
