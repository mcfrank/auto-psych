"""Population mixture of literal and pragmatic listener types.

The participant population is heterogeneous: literal listeners select referents
based on semantic truth and baseline salience without mentalizing the speaker,
while pragmatic listeners invert an informative speaker to make decisive choices.
Observed choices reflect the population mixture of these two listener strategies.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "p_pragmatic": dist.Beta(1.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "w_valence": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        (params["w_features"] + params["w_valence"] * ctx.valence) * ctx.feature_count
        + params["w_familiar"] * ctx.familiarization
    )
    l0_heard = L0(ctx.lex, prior)[ctx.utterance]
    l1_heard = L1(params["alpha"], params["beta"], ctx.lex, prior)[ctx.utterance]
    heard = params["p_pragmatic"] * l1_heard + (1.0 - params["p_pragmatic"]) * l0_heard

    prior_prag = softmax_prior(params["beta"] * jnp.log(prior + EPS))
    prior_pop = params["p_pragmatic"] * prior_prag + (1.0 - params["p_pragmatic"]) * prior
    return with_lapse(jnp.where(ctx.is_prior > 0, prior_pop, heard), params["lapse"])
