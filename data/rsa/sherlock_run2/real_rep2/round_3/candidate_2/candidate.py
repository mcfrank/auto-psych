"""Non-Bayesian heuristic listener selecting the referent with the fewest features.

Rather than recursively simulating counterfactual utterances via Bayes' rule, the
listener employs a direct parsimony heuristic: among candidate objects that literally
match the heard word, the listener chooses with probability inversely weighted by the
object's total feature count, favoring referents without extraneous unmentioned features.
On prior trials without an informative word, choice is uniform guessing. A lapse parameter
mixes in random choice.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def H[u: UTT, r: OBJ](beta, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * vec(feature_count, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = H(params["beta"], ctx.lex, ctx.feature_count)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
