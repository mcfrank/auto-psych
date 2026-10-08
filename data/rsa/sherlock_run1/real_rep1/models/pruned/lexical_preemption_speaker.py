"""RSA pragmatic listener inverting a speaker constrained by lexical preemption.

When an intended referent possesses a dedicated unique descriptor in the display,
speakers incur a penalty for producing an ambiguous shared descriptor instead.
Pragmatic listeners invert this preemption constraint, ruling out referents
with unique identifiers upon hearing a shared word.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_preempt": dist.Normal(0.0, 2.0),
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
def L1[u: UTT, r: OBJ](alpha, w_preempt, lex: ..., prior: ..., preempt: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex, prior) + {EPS})
                - w_preempt * at(preempt, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )

    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    feat_freq = jnp.sum(real_lex, axis=1)
    is_unique_feat = (feat_freq == 1.0) & (1.0 - ctx.is_sink > 0)
    has_unique_feat = jnp.any(is_unique_feat[:, None] * (real_lex > 0), axis=0)
    preempt = (has_unique_feat[None, :] & (~is_unique_feat[:, None]) & (real_lex > 0)).astype(jnp.float32)

    heard = L1(params["alpha"], params["w_preempt"], ctx.lex, prior, preempt)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
