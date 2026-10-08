"""Pragmatic listener with valence-modulated perceptual salience prior.

Listeners ground reference resolution in the speaker's subjective valuation
of referents, where prior referential expectations are driven by perceptual
salience (visual feature richness and chromatic prominence) modulated by
the speaker's affective valence. In neutral and positive contexts, listeners
expect speakers to refer to more salient, feature-rich, or colorful objects;
when the speaker specifies a negative attitude ('least favorite'), this
evaluative preference reverses, directing expectations toward plain,
feature-sparse objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 2.0),
    "w_color": dist.Normal(0.0, 2.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_prior(ctx, w_features, w_valence, w_color, w_familiar):
    effective_features = w_features + w_valence * ctx.valence
    color_prominence = 1.0 - ctx.grayscale
    return softmax_prior(
        effective_features * ctx.feature_count
        + w_color * color_prominence
        + w_familiar * ctx.familiarization
    )


def choice_probs(params, ctx):
    prior = compute_prior(
        ctx,
        params["w_features"],
        params["w_valence"],
        params["w_color"],
        params["w_familiar"],
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
