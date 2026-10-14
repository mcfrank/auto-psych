"""Non-Bayesian most-distinctive heuristic listener.

Listeners interpret utterances using a non-Bayesian distinctiveness heuristic:
when hearing a descriptive label, they restrict candidate referents to literal
matches and select the object with the greatest contextual distinctiveness from
other items in the scene, rather than simulating an informative speaker. In the
absence of an informative descriptor on mumble trials, listeners directly
choose the most visually distinctive object across the entire display.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 1.0),
    "gamma": dist.Normal(0.0, 1.0),
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
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    distinct = jnp.sum(total_dist, axis=1)

    heard = L_heuristic(
        params["beta"], ctx.lex, distinct
    )[ctx.utterance]
    prior = softmax_prior(params["gamma"] * distinct)
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
