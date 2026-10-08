"""Listener type mixture: population mixture of literal and pragmatic listeners."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost": dist.Normal(0.0, 1.0),
    "pi_pragmatic": dist.Beta(1.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, cost, lex: ..., prior: ..., utt_cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS}) - cost * vec(utt_cost, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    base_prior = softmax_prior(params["w_familiar"] * ctx.familiarization)
    salience = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    utt_cost = jnp.maximum(0.0, jnp.sum(ctx.lex * (1.0 - ctx.is_sink)[:, None], axis=1) - 1.0)
    literal = L0(ctx.lex, salience)[ctx.utterance]
    pragmatic = L1(params["alpha"], params["cost"], ctx.lex, base_prior, utt_cost)[ctx.utterance]
    heard = params["pi_pragmatic"] * pragmatic + (1.0 - params["pi_pragmatic"]) * literal
    choice = jnp.where(ctx.is_prior > 0, salience, heard)
    return with_lapse(choice, params["lapse"])
