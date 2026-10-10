"""Twin-contrast solitary perceptual oddity listener with visual feature complexity prior.

Refining twin_contrast_solitary_oddity_listener by incorporating visual feature complexity
weighting into the ungrounded prior over candidate referents from rsa_l2_singleton_feat_color_valence_l0.
Listeners interpret referring expressions using a solitary perceptual oddity heuristic where an
odd-one-out pop-out bias operates exclusively when a solitary singleton contrasts against an
identical twin pair rather than majority triplets or clusters. On uninformative prior trials
without an informative word, when visual displays lack twin contrast, listeners expect speakers
to target richer, feature-laden exemplars with greater descriptive complexity; when an
informative referring expression is heard, selective attention to semantic applicability
directs choice among matching referents, preserving solitary oddity pop-out when contrasting
with twins.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_prior": dist.Normal(0.0, 2.0),
    "beta_utt": dist.Normal(0.0, 2.0),
    "w_feat": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](beta_utt, lex: ..., is_singleton: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(beta_utt * vec(is_singleton, r)),
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
    prior = softmax_prior(
        params["beta_prior"] * is_singleton
        + params["w_feat"] * ctx.feature_count
    )
    heard = L_heur(params["beta_utt"], ctx.lex, is_singleton)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
