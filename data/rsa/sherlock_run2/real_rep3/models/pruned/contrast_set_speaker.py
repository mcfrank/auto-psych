"""Pragmatic listener reasoning over a contrast-set-sensitive speaker (Sedivy et al. 1999).

Speakers in reference games actively seek contrastive differentiation: candidate
referring expressions are evaluated by whether they contrastively differentiate the
intended referent from minimal contrast counterparts in the visual context that lack
the named feature. Pragmatic listeners invert this contrast-sensitive speaker across
recursive reasoning levels, resolving reference by expecting speakers to name features
that establish salient visual contrast sets.
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


@memo
def L2[u: UTT, r: OBJ](alpha, w_contrast, lex: ..., contrast: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, w_contrast, lex, contrast) + {EPS})
                + w_contrast * at(contrast, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_contrast_and_prior(ctx, w_contrast):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words

    feats = jnp.vstack([real_lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = jnp.abs(feats[:, :, None] - feats[:, None, :])
    total_dist = jnp.sum(diff, axis=0)

    r_has_u = real_lex[:, :, None]
    r_prime_lacks_u = (1.0 - real_lex)[:, None, :]
    valid_pair = r_has_u * r_prime_lacks_u

    d_other = jnp.maximum(0.0, total_dist[None, :, :] - 1.0)
    contrast_pair = jnp.exp(-d_other) * valid_pair
    contrast = jnp.sum(contrast_pair, axis=2)

    salience = jnp.sum(contrast, axis=0)
    prior = softmax_prior(w_contrast * salience)
    return contrast, prior


def choice_probs(params, ctx):
    contrast, prior = compute_contrast_and_prior(ctx, params["w_contrast"])
    heard = L2(params["alpha"], params["w_contrast"], ctx.lex, contrast)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
