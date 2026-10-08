"""Non-Bayesian heuristic listener that selects the referent with the fewest features.

People interpret referring expressions using an economy-of-features simplicity
heuristic rather than recursive Bayesian mentalizing: when hearing a descriptive
word, listeners restrict attention to matching referents and choose the candidate
with the fewest total features. People heuristically assume that a single uttered
word is intended to identify an unadorned, minimal referent, penalizing referents
proportionally to their extraneous unmentioned features.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "w_prior_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heuristic[u: UTT, r: OBJ](lex: ..., score: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * exp(vec(score, r)))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    score = -params["beta"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    heard = L_heuristic(ctx.lex, score)[ctx.utterance]
    prior = softmax_prior(
        params["w_prior_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
