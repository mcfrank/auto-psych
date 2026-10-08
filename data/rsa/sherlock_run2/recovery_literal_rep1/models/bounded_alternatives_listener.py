"""Bounded alternatives listener: resource-limited listener with limited attention to alternative utterances.

Listeners have bounded attentional resources to generate and reason over
unmentioned alternative utterances when interpreting a speaker. When simulating
the speaker's utterance choice, a resource-limited listener discounts
unmentioned alternative expressions relative to the focal heard word, reducing
the speaker's perceived likelihood of having chosen an alternative descriptor.
Under limited attention to alternatives, the listener relies predominantly on
literal semantics and contextual salience, attenuating the pragmatic pressure of
scalar implicature.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "att_alt": dist.Beta(1.0, 1.0),
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
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., att: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * vec(att, u)
            * exp(alpha * log(L0[u, r](lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    n_utt = ctx.lex.shape[0]
    is_heard = jnp.arange(n_utt) == ctx.utterance
    is_alt = (1.0 - is_heard.astype(jnp.float32)) * (1.0 - ctx.is_sink)
    att = jnp.where(is_alt > 0.0, params["att_alt"], 1.0)
    heard = L1(params["alpha"], ctx.lex, prior, att)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
