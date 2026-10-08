
"""Pragmatic listener reasoning about a speaker with feature-mapping uncertainty.

When producing a referring expression, the speaker selects an informative feature
of the intended referent to convey, and chooses a word to express that feature.
However, communication is subject to lexical feature uncertainty: while the word
conventionally matches the intended feature, with probability gamma an alternative
feature word is produced due to lexical confusion or transmission noise.
The pragmatic listener inverts this generative model, resolving reference by
marginalizing over the latent intended feature.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.Beta(1.0, 9.0),
}

@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]

@memo
def L1[u: UTT, r: OBJ](alpha, feature_weights: ..., lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            f in UTT,
            wpp=at(lex, f, r) * exp(alpha * log(L0[f, r](lex) + {EPS})),
        ),
        speaker: chooses(u in UTT, wpp=at(feature_weights, u, f)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]

def choice_probs(params, ctx):
    is_real = 1.0 - ctx.is_sink
    eye = jnp.eye(ctx.is_sink.shape[0])
    feature_weights = jnp.where(
        eye > 0.5,
        1.0,
        jnp.where(is_real[:, None] > 0.5, params["gamma"], 0.0),
    )
    heard = L1(params["alpha"], feature_weights, ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return jnp.where(ctx.is_prior > 0, uniform, heard)
