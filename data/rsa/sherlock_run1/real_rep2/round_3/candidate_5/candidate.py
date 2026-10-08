"""RSA pragmatic listener (depth 1) with diagnostic contrast, color salience, and costly speaker.

Refines costly_speaker_contrast_prior by incorporating perceptual color
distinctiveness into the common-knowledge object salience prior (component
taken from rsa_l2_costly_color_salience). In visual reference displays, distinctive
chromatic coloration heightens an object's visual pop-out and prior communicative
salience relative to grayscale competitors, alongside unique semantic feature
contrast and familiarization base rates, while speakers evaluate candidate
utterances by balancing literal informativeness against utterance ambiguity costs.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_contrast": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "w_color": dist.Normal(0.0, 1.0),
    "cost_ambiguity": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex, prior) + {EPS}) - vec(cost, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    non_sink_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    n_u = jnp.sum(non_sink_lex, axis=-1, keepdims=True)
    is_unique = jnp.where(n_u == 1.0, 1.0, 0.0)
    is_shared = jnp.where(n_u > 1.0, 1.0, 0.0)
    contrast = jnp.sum(non_sink_lex * is_unique, axis=0) - jnp.sum(
        non_sink_lex * is_shared, axis=0
    )
    prior = softmax_prior(
        params["w_contrast"] * contrast
        + params["w_familiar"] * ctx.familiarization
        + params["w_color"] * (1.0 - ctx.grayscale)
    )
    n_obj_true = jnp.sum(ctx.lex * (1.0 - ctx.is_sink)[:, None], axis=-1)
    ambiguity = jnp.maximum(0.0, n_obj_true - 1.0)
    cost = params["cost_ambiguity"] * ambiguity
    heard = L1(params["alpha"], ctx.lex, prior, cost)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
