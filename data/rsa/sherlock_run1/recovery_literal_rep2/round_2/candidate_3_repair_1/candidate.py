"""Pragmatic listener with soft literal truth conditions.

The simulated literal listener interprets referring expressions with soft truth
conditions, where absent features retain residual semantic applicability. The
speaker communicates rationally using true feature expressions while
anticipating this soft literal interpretation, which naturally bounds the
speaker's utility differences. The pragmatic listener inverts this rational
speaker to infer the intended referent.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "logit_epsilon": dist.Normal(1.0, 0.5),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](soft_lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(soft_lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., soft_lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](soft_lex, prior) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    eps = jax.nn.sigmoid(params["logit_epsilon"])
    soft_lex = jnp.where(
        ctx.is_sink[:, None] > 0,
        ctx.lex,
        (1.0 - eps) * ctx.lex + eps * (1.0 - ctx.lex),
    )
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, soft_lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
