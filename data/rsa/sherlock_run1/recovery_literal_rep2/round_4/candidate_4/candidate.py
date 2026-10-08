"""Non-Bayesian heuristic listener penalizing contextual feature rarity.

Refines fewest_features_listener by replacing raw feature counts with contextual
feature rarity: listeners penalize candidate referents not for total visual complexity,
but specifically for possessing extraneous features that are distinctive and rare
in the visual scene. Unmentioned features that could have uniquely identified a competitor
incur a heavy penalty, while widely shared features incur minimal penalty.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heuristic[u: UTT, r: OBJ](beta, lex: ..., rarity_score: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * vec(rarity_score, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_utt = (1.0 - ctx.is_sink)[:, None] * ctx.lex
    n_referents = jnp.sum(real_utt, axis=-1)
    rarity_weight = jnp.where(ctx.is_sink > 0, 0.0, 1.0 / (n_referents + EPS))
    rarity_score = jnp.sum(real_utt * rarity_weight[:, None], axis=0)

    heard = L_heuristic(params["beta"], ctx.lex, rarity_score)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
