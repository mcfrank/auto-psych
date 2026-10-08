"""RSA depth-1 listener with contextual distinctiveness prior over objects.

Listeners evaluate candidate referents according to contextual distinctiveness
rather than raw feature count. Features that are shared across all objects
provide minimal contrast, whereas rare or unique features pop out visually
and informatively. In pragmatic reference resolution, listeners reason about
a speaker whose prior over referents scales with contextual distinctiveness,
favoring objects with distinctive features over those with ubiquitous ones.
Prior-elicitation trials reflect uniform guessing. A lapse parameter mixes
in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    features = (1.0 - ctx.is_sink)[:, None] * ctx.lex
    freq = features.sum(axis=1)
    safe_freq = jnp.where(freq > 0.0, freq, 1.0)
    distinctiveness = (features / safe_freq[:, None]).sum(axis=0)

    prior = softmax_prior(
        params["w_distinct"] * distinctiveness + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
