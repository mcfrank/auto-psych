"""Pragmatic listener combining Bayesian salience, distractor suppression, and costly ambiguous speech.

Refines distractor_bayesian_salience_listener by incorporating utterance ambiguity costs
(from costly_feature_speaker): speakers pay an effort cost for producing ambiguous words shared
across multiple referents in the visual context, disincentivizing non-distinguishing descriptions.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "salience_weight": dist.Normal(0.0, 1.0),
    "base_rate_weight": dist.LogNormal(0.0, 1.0),
    "distractor_attention": dist.Beta(2.0, 2.0),
    "cost": dist.HalfNormal(1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., obj_attention: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * (vec(obj_attention, r) + {EPS}))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, prior: ..., lex: ..., obj_attention: ..., costs: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * (log(L0[u, r](lex, obj_attention) + {EPS}) - vec(costs, u))),
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
    matches_heard = ctx.lex[ctx.utterance]
    attention = jnp.where(matches_heard > 0, 1.0, params["distractor_attention"])
    attention = jnp.where(ctx.is_prior > 0, 1.0, attention)

    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    extension = jnp.sum(real_lex, axis=1)
    costs = params["cost"] * jnp.maximum(0.0, extension - 1.0)

    heard = L1(params["alpha"], prior, ctx.lex, attention, costs)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
