"""Attended display listener with decision rationality beta.

Refines attended_display_listener by incorporating a power-law choice rule governed
by decision rationality parameter beta (taken from rsa_l1_rational_referential_salience).
Listeners focus visual attention on candidate referents matching the description,
attenuating non-matching distractor objects outside the attentional spotlight when
simulating the speaker. The listener then selects candidate referents by scaling posterior
probabilities by exponent beta rather than strict probability matching, moderating
extreme inferential bets across forced-choice displays. On prior-elicitation trials with
no communicative word, listeners fall back to uniform guessing. A lapse parameter mixes
in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "att_distractor": dist.Beta(1.0, 3.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., attention: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * vec(attention, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ..., attention: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, attention, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    matching = ctx.lex[ctx.utterance]
    att = matching + params["att_distractor"] * (1.0 - matching)
    attention = jnp.where(ctx.is_prior > 0, jnp.ones_like(att), att)

    heard = L1(params["alpha"], params["beta"], ctx.lex, attention, prior)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
