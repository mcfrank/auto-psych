"""Pragmatic listener with a valence-aligned referent desirability prior.

Speakers evaluate candidate referents based on visual desirability, combining
semantic feature count and chromatic vibrancy (color vs grayscale). In neutral
and positive ('favorite') contexts, speakers favor visually rich objects, whereas
under negative ('least favorite') framing, this preference is inverted toward
plain, unadorned objects. The pragmatic listener inverts this valenced speaker.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
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
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_referent_prior(ctx, w_features, w_color):
    """Compute referent salience prior modulated by evaluative framing and color."""
    stance = jnp.where(ctx.valence == 0.0, 0.5, ctx.valence)
    richness = w_features * ctx.feature_count + w_color * (1.0 - ctx.grayscale)
    return softmax_prior(stance * richness)


def choice_probs(params, ctx):
    prior = compute_referent_prior(ctx, params["w_features"], params["w_color"])
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    chosen = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(chosen, params["lapse"])
