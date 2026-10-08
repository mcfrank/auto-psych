"""Pragmatic listener with selective visual attention to display referents.

Hypothesis: Listeners have limited visual attention across the display,
selectively focusing on candidate referents compatible with the heard
utterance while discounting unattended distractors in the visual background.
When evaluating the speaker's communicative alternatives, the listener computes
informativeness primarily over the attended candidates, so non-matching
distractors exert attenuated influence on pragmatic inference. In the absence
of an informative utterance, the listener allocates attention uniformly
across the available objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "distractor_attention": dist.Beta(1.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., att: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(att, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., att: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, att) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    matching = ctx.lex[ctx.utterance]
    att = jnp.where(matching > 0, 1.0, params["distractor_attention"])
    heard = L1(params["alpha"], ctx.lex, att)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
