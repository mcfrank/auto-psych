"""Depth-1 RSA with shared salience prior, softmax listener, and silent alternative speaker.

Refines softmax_listener_shared_prior by allowing the speaker to choose silence
(the sink utterance) with an explicit cost of silence, rather than only speaking
true feature words. The pragmatic listener applies softmax decision rationality
when selecting referents while maintaining common knowledge of salience.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "cost_silence": dist.Normal(0.0, 2.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, cost_silence, lex: ..., prior: ..., is_sink: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=(1.0 - vec(is_sink, u)) * at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS}))
            + vec(is_sink, u) * exp(-cost_silence),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(
        params["alpha"],
        params["beta"],
        params["cost_silence"],
        ctx.lex,
        prior,
        ctx.is_sink,
    )[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
