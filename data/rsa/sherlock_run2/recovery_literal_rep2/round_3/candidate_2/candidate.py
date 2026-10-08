"""Distinctiveness heuristic listener: non-Bayesian reference resolution via contextual distinctiveness.

Rather than recursively simulating a counterfactual speaker, the listener applies a direct
perceptual distinctiveness heuristic. When an utterance describes an object, the listener
selects among matching referents by choosing the object that stands out as most distinctive
in the visual array, possessing features that are rarest across the display. On prior-elicitation
trials, choices are guided by this same distinctiveness heuristic alongside familiarization base
rates. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heuristic[u: UTT, r: OBJ](beta, lex: ..., distinct: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(beta * vec(distinct, r)),
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
    heard = L_heuristic(params["beta"], ctx.lex, distinctiveness)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
