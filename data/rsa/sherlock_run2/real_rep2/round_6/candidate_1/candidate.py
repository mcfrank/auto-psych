"""Pragmatic listener with feature-based visual attention bottleneck.

Hearing a referring expression directs the listener's attentional focus to the
named visual attribute, attenuating attention to unmentioned alternative features
across the display. When simulating what the speaker could have said, the listener
perceives unmentioned alternative descriptions with reduced attentional availability,
scaling down the speaker's communicative pressure to use alternative words. On
uninformative prior trials without an uttered word, visual attention is uniformly
distributed across all features. Listeners reason at depth 2. A lapse parameter
mixes in uniform guessing.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "logit_att": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., att: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=vec(att, u) * at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., att: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=vec(att, u) * at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, att) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    n_utt = ctx.lex.shape[0]
    att_alt = jax.nn.sigmoid(params["logit_att"])
    is_heard = jnp.arange(n_utt) == ctx.utterance
    att = jnp.where(ctx.is_prior > 0, 1.0, jnp.where(is_heard, 1.0, att_alt))
    heard = L2(params["alpha"], ctx.lex, att)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
