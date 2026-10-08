"""Valence distinctiveness listener: distinctiveness heuristic modulated by evaluative framing.

Refines distinctiveness_heuristic_listener by incorporating communicative valence framing
(taken from evaluative_prominence_listener). When interpreting referring expressions under
neutral or positive framing, listeners penalize matching referents that possess extraneous
distinctive features; under evaluative framing ('least favorite'), this heuristic inverts
so that listeners prefer the referent with higher contextual distinctiveness.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heuristic[u: UTT, r: OBJ](coeff, lex: ..., distinct: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(coeff * vec(distinct, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    features = (1.0 - ctx.is_sink)[:, None] * ctx.lex
    freq = features.sum(axis=1)
    safe_freq = jnp.where(freq > 0.0, freq, 1.0)
    distinctiveness = (features / safe_freq[:, None]).sum(axis=0)

    prior = softmax_prior(
        params["beta"] * distinctiveness + params["w_familiar"] * ctx.familiarization
    )
    coeff = params["beta"] - params["w_valence"] * ctx.valence
    heard = L_heuristic(coeff, ctx.lex, distinctiveness)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
