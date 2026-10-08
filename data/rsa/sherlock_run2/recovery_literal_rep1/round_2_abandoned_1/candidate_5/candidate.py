"""Literal parsimonious listener (depth 0): literal choice with descriptive parsimony penalty.

Extends literal_salience_listener by incorporating a descriptive parsimony mechanism:
listeners interpret utterances literally without simulating alternative words, but discount
referents possessing surplus unmentioned features. On prior trials with no informative word,
choice is guided solely by contextual salience. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "w_excess": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](w_excess, lex: ..., prior: ..., excess: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r) * exp(-w_excess * vec(excess, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    excess = jnp.maximum(0.0, ctx.feature_count - 1.0)
    heard = L0(params["w_excess"], ctx.lex, prior, excess)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
