"""Recursive contrastive foil speaker model.

Rather than communicating to uniquely identify a referent among all objects in the scene,
the speaker's communicative goal is contrastive: to distinguish the intended referent
from a specific competitor foil. Speakers choose an intended referent and a competitor
foil to eliminate, uttering a word that resolves that binary contrast. Pragmatic listeners
reason recursively at depth 2 by inverting a speaker who anticipates a depth-1 contrastive
listener, determining the intended referent by marginalizing over latent competitor foils.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(c in OBJ, wpp=(r != c)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * log(
                    at(lex, u, r)
                    / (at(lex, u, r) + at(lex, u, c) + {EPS})
                    + {EPS}
                )
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(c in OBJ, wpp=(r != c)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * log(
                    L1[u, r](alpha, lex)
                    / (L1[u, r](alpha, lex) + L1[u, c](alpha, lex) + {EPS})
                    + {EPS}
                )
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    heard = L2(params["alpha"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
