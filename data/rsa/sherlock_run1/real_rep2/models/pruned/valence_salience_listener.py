"""Valence-salience listener: affective framing modulates visual salience in pragmatic reference.

Listeners interpret referring expressions by modeling the speaker as selecting
referents according to valence-modulated visual salience: an object's prior communicative
probability reflects its visual richness (features and color) scaled by the speaker's
affective framing. Under positive or default neutral framing, feature-rich and colored
objects are favored referents, whereas under negative framing ('least favorite'), this
preference inverts so that minimal, unadorned objects become the primary expected referents.
Pragmatic listeners invert this evaluative speaker to guide referent selection both when
interpreting informative words and when guessing without a description.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 2.0),
    "w_color": dist.Normal(0.0, 1.0),
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
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    eff_feature_weight = params["w_features"] + params["w_valence"] * ctx.valence
    salience = (
        eff_feature_weight * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
        + params["w_familiar"] * ctx.familiarization
    )
    prior = softmax_prior(salience)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
