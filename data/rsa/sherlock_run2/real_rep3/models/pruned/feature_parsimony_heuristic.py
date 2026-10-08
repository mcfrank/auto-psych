"""Non-Bayesian feature-parsimony heuristic listener.

Rather than simulating a recursive speaker via Theory of Mind, the listener
interprets referring expressions through a direct heuristic of feature parsimony:
among objects that literally possess the heard feature, the listener penalizes
referents that possess extraneous unmentioned features, exponentially favoring
the most economical referent (in line with Shepard's law of generalization and
Gricean minimality). With no informative word, choices are uniform. A lapse
parameter captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](beta, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * exp(-beta * vec(feature_count, r)))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L_heur(params["beta"], ctx.lex, ctx.feature_count)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
