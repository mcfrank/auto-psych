"""Parsimonious literal mixture: population mixture of literal and parsimonious listeners."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.HalfNormal(1.0),
    "pi_parsimonious": dist.Beta(2.0, 2.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L_parsimonious[u: UTT, r: OBJ](lex: ..., weight: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(weight, r))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(params["w_familiar"] * ctx.familiarization)
    weight = jnp.exp(-params["beta"] * ctx.feature_count)
    literal = L0(ctx.lex)[ctx.utterance]
    parsimonious = L_parsimonious(ctx.lex, weight)[ctx.utterance]
    heard = params["pi_parsimonious"] * parsimonious + (1.0 - params["pi_parsimonious"]) * literal
    choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(choice, params["lapse"])
