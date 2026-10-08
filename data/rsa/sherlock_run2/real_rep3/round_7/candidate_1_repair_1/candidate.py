"""Lexical preemption heuristic listener.

Listeners interpret referring expressions using a non-Bayesian lexical
preemption heuristic rather than recursive Theory of Mind. When hearing a word
that applies to multiple referents, listeners choose among candidate objects
possessing that feature by penalizing candidates that possess unmentioned
distinguishing features that could have uniquely identified them in the context,
favoring referents that lack an exclusive alternative label. On uninformative
trials without an informative word, listeners choose uniformly among available
objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](beta, lex: ..., alt_dist: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * exp(-beta * at(alt_dist, u, r)))
    return Pr[listener.r == r]


def compute_alt_distinguishing(ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    extension = jnp.sum(real_lex, axis=1)
    is_dist = (extension == 1.0).astype(jnp.float32)
    dist_per_obj = jnp.sum(real_lex * is_dist[:, None], axis=0)
    is_u_dist = is_dist[:, None] * real_lex
    return dist_per_obj[None, :] - is_u_dist


def choice_probs(params, ctx):
    alt_dist = compute_alt_distinguishing(ctx)
    heard = L_heur(params["beta"], ctx.lex, alt_dist)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
