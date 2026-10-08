"""Attended display listener: reference resolution under limited visual attention to the display.

When an utterance is heard, the listener focuses visual attention on candidate
referents matching the description, attenuating non-matching distractor objects
outside the attentional spotlight. The listener's simulated speaker evaluates the
communicative utility of alternative utterances with respect to this attended
display rather than the unweighted visual scene.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "att_distractor": dist.Beta(1.0, 3.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., attention: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(attention, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., attention: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, attention) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    # Attentional weight across objects: matching objects receive full attention (1.0),
    # while non-matching distractor objects receive attenuated attention (att_distractor).
    matching = ctx.lex[ctx.utterance]
    att = matching + params["att_distractor"] * (1.0 - matching)
    attention = jnp.where(ctx.is_prior > 0, jnp.ones_like(att), att)

    heard = L1(params["alpha"], ctx.lex, attention, prior)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
