"""RSA pragmatic listener at depth 1 with Bayesian log-odds base-rate integration.

Refines bayesian_base_rate_l2 by reducing recursion depth from depth 2 to depth 1:
listeners invert a softmax-rational speaker reasoning over a literal listener (L1),
combining visual feature complexity with empirical familiarization base rates via
Bayesian log-odds evidence integration.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "salience_weight": dist.Normal(0.0, 1.0),
    "base_rate_weight": dist.LogNormal(0.0, 1.0),
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
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    uniform = jnp.full_like(ctx.feature_count, 1.0 / ctx.feature_count.shape[0])
    safe_fam = jnp.where(ctx.has_familiarization > 0, ctx.familiarization, uniform)
    fam_log_evidence = jnp.where(
        ctx.has_familiarization > 0,
        params["base_rate_weight"] * jnp.log(safe_fam + EPS),
        0.0,
    )
    log_prior = params["salience_weight"] * ctx.feature_count + fam_log_evidence
    prior = softmax_prior(log_prior)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
