"""Pragmatic listener with limited visual attention to display distractors.

Hearing a referring expression restricts the listener's attentional spotlight
to candidate referents that match the uttered word. Distractor objects that lack
the uttered word receive attenuated visual attention. When evaluating why the
speaker chose that word, the listener simulates a speaker whose communicative
success is evaluated against this attended context, discounting unattended
distractors. On prior trials without an informative word, attention is uniformly
distributed. A lapse parameter mixes in uniform guessing.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "logit_att": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., att: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(att, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., att: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, att) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    att_distractor = jax.nn.sigmoid(params["logit_att"])
    is_cand = jnp.where(ctx.is_prior > 0, 1.0, ctx.lex[ctx.utterance]) > 0
    att = jnp.where(is_cand, 1.0, att_distractor)
    heard = L1(params["alpha"], ctx.lex, att)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
