"""Twin-contrast solitary perceptual oddity listener with contextual distinctiveness.

Refining twin_contrast_solitary_oddity_listener by incorporating the continuous
contextual distinctiveness component from rsa_l2_singleton_feat_color_valence_l0.
Listeners interpret referring expressions by integrating a solitary perceptual oddity
heuristic with continuous visual distinctiveness. While the solitary oddity indicator
triggers an odd-one-out pop-out bias exclusively when a solitary singleton contrasts
against an identical twin pair, continuous contextual distinctiveness additionally
weights candidate referents by their continuous feature contrast (Hamming distance)
relative to all other items in the visual display. In both ungrounded prior expectations
and semantic utterance comprehension, referents are weighted by both twin-contrast
solitary oddity pop-out and continuous contextual distinctiveness, defaulting to uniform
choice on contexts with identical or feature-symmetric displays.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_prior": dist.Normal(0.0, 2.0),
    "beta_utt": dist.Normal(0.0, 2.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](
    beta_utt, w_distinct, lex: ..., is_singleton: ..., distinct: ...
):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * exp(
            beta_utt * vec(is_singleton, r)
            + w_distinct * vec(distinct, r)
        ),
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


def compute_distinct(ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    return jnp.sum(diff, axis=(0, 2))


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    distinct = compute_distinct(ctx)
    prior = softmax_prior(
        params["beta_prior"] * is_singleton
        + params["w_distinct"] * distinct
    )
    heard = L_heur(
        params["beta_utt"],
        params["w_distinct"],
        ctx.lex,
        is_singleton,
        distinct,
    )[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
