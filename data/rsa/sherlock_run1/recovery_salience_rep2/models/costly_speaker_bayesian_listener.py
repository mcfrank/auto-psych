"""Pragmatic listener combining Bayesian salience and base rates with a costly speaker.

Refines bayesian_salience_base_rate_listener by incorporating utterance ambiguity costs
(from costly_feature_speaker): speakers pay a cost for producing ambiguous words shared
across multiple referents in the display, disincentivizing non-distinguishing descriptions.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "salience_weight": dist.Normal(0.0, 1.0),
    "base_rate_weight": dist.LogNormal(0.0, 1.0),
    "cost": dist.HalfNormal(1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, prior: ..., lex: ..., costs: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * (log(L0[u, r](lex) + {EPS}) - vec(costs, u))),
        ),
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
    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    extension = jnp.sum(real_lex, axis=1)
    costs = params["cost"] * jnp.maximum(0.0, extension - 1.0)
    heard = L1(params["alpha"], prior, ctx.lex, costs)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
