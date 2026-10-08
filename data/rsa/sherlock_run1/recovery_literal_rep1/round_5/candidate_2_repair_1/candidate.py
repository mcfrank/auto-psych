"""Pragmatic listener inverting a competitor-contrast-cost speaker.

Speakers incur a cognitive production cost when producing an utterance that
discriminates the intended referent from its most confusable competitor in
the visual scene, penalizing contrastive words relative to shared labels.
The listener inverts this production process with decision rationality alpha,
with lapses defaulting to perceptual salience.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "c_contrast": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, utt_weights: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(utt_weights, u, r)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(alpha * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    real = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real[:, None]
    n_obj = ctx.lex.shape[1]
    eye = jnp.eye(n_obj)
    distractor_mask = 1.0 - eye

    # Overlap of features between objects (how confusable they are)
    overlap = (real_lex.T @ real_lex) * distractor_mask
    comp_denom = jnp.sum(overlap, axis=1, keepdims=True)
    uniform_comp = distractor_mask / jnp.maximum(n_obj - 1.0, 1.0)
    comp_weights = jnp.where(comp_denom > 0, overlap / (comp_denom + EPS), uniform_comp)

    # Contrast value: does utterance u rule out the confusable competitors of r?
    contrast = (1.0 - ctx.lex) @ comp_weights.T
    utt_weights = ctx.lex * jnp.exp(-params["c_contrast"] * contrast)

    heard = L1(params["alpha"], utt_weights, prior)[ctx.utterance]
    choice = (1.0 - params["lapse"]) * heard + params["lapse"] * prior
    return jnp.where(ctx.is_prior > 0, prior, choice)
