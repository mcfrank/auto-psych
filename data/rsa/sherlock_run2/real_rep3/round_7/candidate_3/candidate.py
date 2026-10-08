"""Pragmatic listener reasoning over a speaker with distinctive aspect communicative goals.

Rather than intending to uniquely transmit an object's spatial token identity,
the speaker selects a communicative goal (Question Under Discussion) targeting a
distinctive aspect of the intended referent, and chooses an utterance to inform
the listener about that aspect. The pragmatic listener inverts this generative
model, jointly inferring the speaker's communicative goal and the intended
referent upon hearing an utterance. On uninformative trials without a word,
referent expectations reflect the contextual communicative affordance of each
object under the speaker's goal distribution.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_goal": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., goal_weights: ..., l0_aspect: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            q in UTT,
            u in UTT,
            wpp=at(goal_weights, q, r)
            * at(lex, u, r)
            * exp(alpha * log(at(l0_aspect, u, q) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_goal_weights_and_prior(ctx, w_goal):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    counts = jnp.sum(real_lex, axis=1)
    distinctiveness = (1.0 / jnp.maximum(counts, 1.0)) * (1.0 - ctx.is_sink)

    goal_scores = w_goal * distinctiveness
    goal_weights = ctx.lex * jnp.exp(goal_scores)[:, None]

    affordance = jnp.sum(real_lex * distinctiveness[:, None], axis=0)
    prior = softmax_prior(w_goal * affordance)
    return goal_weights, prior


def choice_probs(params, ctx):
    l0_r = L0(ctx.lex)
    l0_q = l0_r @ ctx.lex.T

    goal_weights, prior = compute_goal_weights_and_prior(ctx, params["w_goal"])
    l1_matrix = L1(params["alpha"], ctx.lex, goal_weights, l0_q, prior)
    heard = l1_matrix[ctx.utterance]
    probs = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(probs, params["lapse"])
