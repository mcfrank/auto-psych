"""Semantic fallback listener.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "semantic_lapse": dist.Beta(1.0, 6.0),
    "lapse": dist.Beta(1.0, 19.0),
}

@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]

@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]

def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    pragmatic = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    literal = L0(ctx.lex, prior)[ctx.utterance]
    heard = (1.0 - params["semantic_lapse"]) * pragmatic + params["semantic_lapse"] * literal
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    choice = jnp.where(ctx.is_prior > 0, uniform, heard)
    return with_lapse(choice, params["lapse"])
