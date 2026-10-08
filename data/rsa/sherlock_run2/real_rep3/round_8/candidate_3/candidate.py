"""Pragmatic listener reasoning about a worst-case competitor-averse speaker.

Speakers in reference games actively avoid communicative risk by penalizing referring
expressions that create an acute danger of misunderstanding, specifically penalizing
candidate utterances in proportion to the maximum probability the listener assigns to
any competing distractor. Pragmatic listeners invert this worst-case-competitor-averse
speaker, resolving reference by expecting speakers to choose words that leave no single
competitor with high probability.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_rival": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, w_rival, lex: ..., worst_rival: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, lex) + {EPS})
                - w_rival * at(worst_rival, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_worst_rival(l1_probs):
    n_obj = l1_probs.shape[1]
    eye = jnp.eye(n_obj)
    l1_masked = jnp.where(eye[None, :, :] > 0.0, -1.0, l1_probs[:, None, :])
    return jnp.maximum(jnp.max(l1_masked, axis=-1), 0.0)


def choice_probs(params, ctx):
    l1_probs = L1(params["alpha"], ctx.lex)
    worst_rival = compute_worst_rival(l1_probs)
    heard = L2(params["alpha"], params["w_rival"], ctx.lex, worst_rival)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
