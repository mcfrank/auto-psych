"""Pragmatic listener with private salience prior inverting a distractor-cost-sensitive naive speaker.

Refines naive_speaker_salience_listener: when a naive speaker produces descriptive
features for an intended referent, they incur an utterance production cost
proportional to the number of non-referent distractors excluded by that feature.
The listener inverts this distractor-cost-sensitive naive production speaker and
scales the resulting evidence by their own perceptual salience prior over objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "c_feature": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., costs: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(-vec(costs, u))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=vec(prior, r) * exp(alpha * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    ext = jnp.sum(ctx.lex, axis=1)
    costs = params["c_feature"] * (1.0 - ctx.is_sink) * (ctx.lex.shape[1] - ext)
    heard = L1(params["alpha"], ctx.lex, costs, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
