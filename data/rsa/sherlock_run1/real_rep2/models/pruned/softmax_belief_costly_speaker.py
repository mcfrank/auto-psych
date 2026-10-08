"""Softmax belief listener with cost-sensitive speaker.

Refines softmax_belief_listener by incorporating utterance ambiguity costs into
the simulated speaker's utility (component taken from costly_speaker_shared_prior):
when multiple candidate referring expressions are literally true of the intended
referent, speakers incur a production cost proportional to the expression's extension
across competitor objects. Listeners invert this cost-sensitive speaker and softly
maximize over the resulting pragmatic posterior beliefs using a softmax decision rule
governed by decision rationality gamma.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "cost_ambiguity": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, gamma, lex: ..., prior: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex, prior) + {EPS}) - vec(cost, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(gamma * Pr[speaker.r == r]))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    n_obj_true = jnp.sum(ctx.lex * (1.0 - ctx.is_sink)[:, None], axis=-1)
    ambiguity = jnp.maximum(0.0, n_obj_true - 1.0)
    cost = params["cost_ambiguity"] * ambiguity
    heard = L1(params["alpha"], params["gamma"], ctx.lex, prior, cost)[ctx.utterance]
    prior_choice = jax.nn.softmax(params["gamma"] * prior)
    return with_lapse(jnp.where(ctx.is_prior > 0, prior_choice, heard), params["lapse"])
