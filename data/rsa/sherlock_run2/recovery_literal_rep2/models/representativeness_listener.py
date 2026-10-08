"""Representativeness listener: reference resolution via diagnostic likelihood ratios.

Listeners resolve referring expressions by evaluating how representative or diagnostic
each candidate referent is for the spoken description, rather than simulating recursive
mental states or applying ungrounded feature penalties. The representativeness of a
referent r for an utterance u is determined by the likelihood ratio comparing the probability
that a speaker intending r would produce u against the probability that a speaker intending
an alternative distractor in the context would produce u. On prior trials where no informative
word is spoken, choice is guided by each object's expected representativeness across its features.
A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "gamma": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_rep[u: UTT, r: OBJ](gamma, lex: ..., log_rep: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r) * exp(gamma * at(log_rep, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    n_obj = ctx.lex.shape[1]

    # Naive speaker production probability P(u | r)
    feat_count = jnp.maximum(1.0, ctx.lex.sum(axis=0))
    p_u_r = ctx.lex / feat_count[None, :]

    # Distractor background probability P(u | not r)
    sum_p = p_u_r.sum(axis=1, keepdims=True)
    p_u_not_r = (sum_p - p_u_r) / jnp.maximum(1.0, n_obj - 1.0)

    # Diagnostic likelihood ratio with baseline distractor rate
    eps0 = 0.05
    rep = (p_u_r + EPS) / (p_u_not_r + eps0)
    log_rep = jnp.log(rep)

    # Prior representativeness: expected log representativeness of the object
    prior_rep = (p_u_r * log_rep).sum(axis=0)

    prior = softmax_prior(
        params["gamma"] * prior_rep + params["w_familiar"] * ctx.familiarization
    )

    heard = L_rep(params["gamma"], ctx.lex, log_rep, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
