"""Pragmatic listener reasoning about a lexical-accessibility-biased speaker.

When selecting referring expressions, speakers balance communicative informativeness
against cognitive lexical accessibility, experiencing lower production cost for
common, high-prevalence features that appear frequently across objects in the
visual scene. Candidate words receive an accessibility advantage proportional to
their contextual extension. Pragmatic listeners invert this accessibility-sensitive
speaker at depth 2, mitigating over-confident pragmatic inferences when a common
word is used.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_access": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., access: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                + vec(access, u)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., access: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, lex, access) + {EPS})
                + vec(access, u)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_lexical_accessibility(lex: jnp.ndarray, is_sink: jnp.ndarray, w_access) -> jnp.ndarray:
    """Compute accessibility advantage of each utterance based on context prevalence."""
    real_words = 1.0 - is_sink
    n_u = jnp.sum(lex * real_words[:, None], axis=1)
    # Prevalence: extra objects beyond singleton (0 for unique words, positive for common words)
    prevalence = jnp.maximum(0.0, n_u - 1.0) * real_words
    return w_access * prevalence


def choice_probs(params, ctx):
    access = compute_lexical_accessibility(ctx.lex, ctx.is_sink, params["w_access"])
    heard = L2(params["alpha"], ctx.lex, access)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
