"""Attenuated solitary perceptual oddity listener with evaluative valence framing.

Refining attenuated_solitary_oddity_listener by incorporating evaluative valence framing
from rsa_l2_singleton_feat_color_valence_l0. Listeners interpret referring expressions
using a solitary perceptual oddity heuristic whose salience is strong during ungrounded
prior expectations but attenuated during linguistic utterance comprehension. Additionally,
listeners modulate their expectations about referent feature complexity according to the
speaker's evaluative framing valence: positive framing ('favorite') biases choice toward
richer, feature-laden exemplars, negative framing ('least favorite') inverts this bias
toward minimal, sparse exemplars, and neutral framing defaults strictly to the solitary
oddity heuristic. A lapse parameter captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_prior": dist.Normal(0.0, 2.0),
    "beta_utt": dist.Normal(0.0, 2.0),
    "w_valence": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](
    beta_utt, lex: ..., is_singleton: ..., valence_bias: ...
):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * exp(
            beta_utt * vec(is_singleton, r)
            + vec(valence_bias, r)
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
    is_solitary_singleton = jnp.where(
        (is_item_singleton > 0.0) & (singleton_count > 0.5) & (singleton_count < 1.5),
        1.0,
        0.0,
    )
    return is_solitary_singleton


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    valence_bias = params["w_valence"] * ctx.valence * ctx.feature_count
    prior = softmax_prior(params["beta_prior"] * is_singleton + valence_bias)
    heard = L_heur(
        params["beta_utt"], ctx.lex, is_singleton, valence_bias
    )[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
