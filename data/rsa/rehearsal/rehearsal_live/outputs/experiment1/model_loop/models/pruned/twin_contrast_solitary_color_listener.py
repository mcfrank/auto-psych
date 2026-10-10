"""Twin-contrast solitary perceptual oddity listener with visual color contrast.

Refining twin_contrast_solitary_oddity_listener by incorporating the visual color
contrast component from rsa_l2_singleton_feat_color_valence_l0. In reference games
where visual displays present candidate referents in full chromatic coloration
alongside desaturated grayscale distractors, listeners exhibit an intrinsic
perceptual bias toward fully colored objects. In both spontaneous ungrounded prior
expectations and linguistic utterance comprehension, candidate referents are
weighted by both twin-contrast solitary oddity pop-out and visual color contrast.
On displays where all objects are uniformly colored, color contrast contributes
identically across items and cancels out under normalization, preserving the
incumbent's twin-contrast oddity heuristic and literal semantic baseline.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_prior": dist.Normal(0.0, 2.0),
    "beta_utt": dist.Normal(0.0, 2.0),
    "w_color": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](beta_utt, lex: ..., is_singleton: ..., color_bias: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * exp(beta_utt * vec(is_singleton, r) + vec(color_bias, r)),
    )
    return Pr[listener.r == r]


def compute_singleton_indicator(ctx):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    is_identical = (dist < 1e-5).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_item_singleton = (copy_count <= 1.0).astype(jnp.float32)
    singleton_count = jnp.sum(is_item_singleton)
    max_copy = jnp.max(copy_count)
    is_solitary_singleton = jnp.where(
        (is_item_singleton > 0.0)
        & (singleton_count > 0.5)
        & (singleton_count < 1.5)
        & (max_copy > 1.5)
        & (max_copy < 2.5),
        1.0,
        0.0,
    )
    return is_solitary_singleton


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    color_bias = params["w_color"] * (1.0 - ctx.grayscale)
    prior = softmax_prior(params["beta_prior"] * is_singleton + color_bias)
    heard = L_heur(params["beta_utt"], ctx.lex, is_singleton, color_bias)[
        ctx.utterance
    ]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
