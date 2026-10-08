"""Valence feature contrast listener: non-recursive pragmatic interpretation with valence-modulated prior.

Refines the incumbent feature_contrast_listener by incorporating affective framing
(taken from valence_salience_listener). Listeners interpret referring expressions
via direct feature contrast (penalizing objects with unmentioned features), but
modulate their object prior based on the speaker's affective framing: positive or
neutral framing favors feature-rich objects, whereas negative framing ('least favorite')
inverts this preference to favor simpler referents.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "theta": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_contrast[u: UTT, r: OBJ](theta, lex: ..., unmentioned: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r) * exp(-theta * at(unmentioned, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    feature_weight = params["w_features"] + params["w_valence"] * ctx.valence
    prior = softmax_prior(
        feature_weight * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    unmentioned = jnp.maximum(0.0, ctx.feature_count[None, :] - ctx.lex)
    heard = L_contrast(params["theta"], ctx.lex, unmentioned, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
