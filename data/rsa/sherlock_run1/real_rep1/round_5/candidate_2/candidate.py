"""RSA pragmatic listener inverting a satisficing speaker who prunes inferior alternatives.

Speakers do not consider all true descriptions of an object with equal baseline
availability; instead, they penalize inferior alternatives in proportion to their
relative informativeness deficit compared to the optimal available descriptor for
that referent. Pragmatic listeners invert this bounded speaker, recognizing that a
speaker will only consider an ambiguous, shared descriptor if the intended referent
lacked a clearly superior alternative.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "kappa": dist.Normal(0.0, 2.0),
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
def L1[u: UTT, r: OBJ](alpha, kappa, lex: ..., prior: ..., deficit: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex, prior) + {EPS})
                - kappa * at(deficit, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def calc_deficit(probs, lex, is_sink):
    is_real = (1.0 - is_sink)[:, None]
    applicable = lex * is_real
    masked_probs = jnp.where(applicable > 0, probs, -1.0)
    max_prob = jnp.maximum(jnp.max(masked_probs, axis=0, keepdims=True), 0.0)
    rel_deficit = jnp.where(
        (applicable > 0) & (max_prob > 0),
        (max_prob - probs) / jnp.maximum(max_prob, 1e-6),
        0.0,
    )
    return jnp.clip(rel_deficit, 0.0, 1.0)


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count
        + params["w_familiar"] * ctx.familiarization
    )
    l0_probs = L0(ctx.lex, prior)
    deficit = calc_deficit(l0_probs, ctx.lex, ctx.is_sink)
    heard = L1(params["alpha"], params["kappa"], ctx.lex, prior, deficit)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
