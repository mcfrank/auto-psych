"""RSA pragmatic listener with contextual visual contrast prior.

An object's prior salience is determined by contextual visual contrast: how
strongly its features stand out against the display background. Ubiquitous
features shared across all objects form the common visual background (zero
contrast), whereas features possessed by fewer competitors generate visual
contrast and pop-out.
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
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_utt = (1.0 - ctx.is_sink)[:, None] * ctx.lex
    n_obj = ctx.lex.shape[-1]
    # Proportion of objects possessing each feature in the context
    feature_prevalence = jnp.sum(real_utt, axis=-1) / n_obj
    # Feature contrast against scene background: (1 - prevalence) for real features
    contrast_weight = jnp.where(ctx.is_sink > 0, 0.0, 1.0 - feature_prevalence)
    # Visual contrast score: sum of contrast weights of features true of each object
    contrast_score = jnp.sum(real_utt * contrast_weight[:, None], axis=0)

    prior = softmax_prior(
        params["w_contrast"] * contrast_score + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
