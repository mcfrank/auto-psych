"""Pragmatic listener reasoning about a contrast-maximizing speaker.

Speakers choose referring expressions to establish contrast against visual
competitors that lack the named feature, rather than merely evaluating target
recovery. Pragmatic listeners invert this contrastive speaker, expecting words
to target referents whose features actively resolve contrast with minimally
different competitors in the visual scene.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_contrast": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_contrast, lex: ..., contrast: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                + w_contrast * at(contrast, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_contrast_and_prior(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words

    feats = jnp.vstack([real_lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    pair_dist = jnp.sum(jnp.abs(diff), axis=0)

    n_obj = pair_dist.shape[0]
    eye = jnp.eye(n_obj)

    # Word-level contrast: competitor r' lacks word u, similarity on other features is exp(1 - pair_dist)
    lacks_u = (1.0 - ctx.lex)[:, None, :]
    sim_other = jnp.exp(1.0 - pair_dist)[None, :, :] * (1.0 - eye)[None, :, :]
    contrast = jnp.sum(lacks_u * sim_other, axis=-1) * real_lex

    # Baseline scene contrast: negative similarity to other objects in display
    sim_objects = jnp.exp(-pair_dist) * (1.0 - eye)
    baseline_contrast = -jnp.sum(sim_objects, axis=1)

    return contrast, baseline_contrast


def choice_probs(params, ctx):
    contrast, baseline_contrast = compute_contrast_and_prior(ctx)
    prior = softmax_prior(params["w_contrast"] * baseline_contrast)
    heard = L1(params["alpha"], params["w_contrast"], ctx.lex, contrast)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
