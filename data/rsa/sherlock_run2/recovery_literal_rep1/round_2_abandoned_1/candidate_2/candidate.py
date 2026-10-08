"""Feature economy listener: bounded reasoning with a penalty for superfluous visual features.

Listeners operate with bounded attentional capacity: upon hearing an utterance,
they focus visual attention strictly on the described feature, penalizing candidate
referents that carry superfluous, unmentioned visual features that create visual clutter.
Rather than recursively simulating alternative utterances a speaker could have chosen,
the listener selects the referent that parsimoniously matches the utterance with minimal
extraneous features.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "penalty_superfluous": dist.HalfNormal(2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    superfluous = jnp.where(
        ctx.is_prior > 0,
        0.0,
        jnp.maximum(0.0, ctx.feature_count - ctx.lex[ctx.utterance]),
    )
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count
        + params["w_familiar"] * ctx.familiarization
        - params["penalty_superfluous"] * superfluous
    )
    heard = L0(ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
