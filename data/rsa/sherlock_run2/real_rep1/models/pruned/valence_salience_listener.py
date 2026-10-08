"""Valence-modulated evaluative salience listener.

Listeners assume speakers choose referents according to an evaluative salience goal:
visual features and chromatic color provide positive baseline appeal, but communicative
framing dictates the polarity of the speaker's preference. Under neutral framing or
when describing a favorite object, the speaker targets feature-rich and colored objects;
when describing a least favorite object, the speaker inverts this preference to target
sparse, featureless objects. Pragmatic listeners invert this valence-modulated speaker
to interpret referring expressions, and choose according to evaluative salience on prior trials.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
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
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    salience_logits = (
        (params["w_features"] + ctx.valence * params["w_valence"])
        * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
    )
    prior = softmax_prior(salience_logits)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
