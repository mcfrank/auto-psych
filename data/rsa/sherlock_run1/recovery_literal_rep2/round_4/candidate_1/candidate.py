"""Pragmatic listener reasoning about a speaker communicating an informative aspect.

Speakers in reference games communicate an informative aspect of the intended
referent rather than its unique identity, selecting a communicative goal from
the object's features with a preference for prominent shared dimensions. When
hearing a referring expression, listeners reason inversely about this communicative
goal, recognizing that an intended referent with multiple competing features
dilutes the speaker's likelihood of mentioning any single one. Listeners select
referents according to their posterior beliefs under a softmax decision rule.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_qud[u: UTT, r: OBJ](alpha, beta, lex: ..., aspect_value: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(g in UTT, wpp=at(lex, g, r) * exp(beta * vec(aspect_value, g))),
        speaker: chooses(u in UTT, to_be=g),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(
        r in OBJ,
        wpp=exp(alpha * log(Pr[speaker.r == r] + {EPS})),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    aspect_count = jnp.sum(ctx.lex, axis=1)
    aspect_value = jnp.log(aspect_count + EPS)
    heard = L_qud(params["alpha"], params["beta"], ctx.lex, aspect_value)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
