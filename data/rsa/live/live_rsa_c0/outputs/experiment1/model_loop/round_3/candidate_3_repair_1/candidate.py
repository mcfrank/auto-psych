"""Aspect QUD listener: pragmatic listener reasoning over referential identification vs descriptive aspect attribution.

Listeners interpret referring expressions by reasoning about a speaker with uncertain
communicative goals: either referential identification (informing a pragmatic listener who
draws scalar inferences) or descriptive aspect attribution (informing a literal listener about
the object's visual features). Under the aspect goal, the speaker selects expressions based on
literal descriptive informativeness rather than recursive scalar contrast, moderating scalar
exclusion and explaining why feature-rich referents retain substantial choice probability on
ubiquitous and scalar displays.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

GOALS = jnp.arange(2)

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "logit_p_ident": dist.Normal(0.5, 1.0),
    "w_singleton": dist.Normal(1.0, 0.5),
    "w_distinct": dist.Normal(0.15, 0.2),
    "w_features": dist.Normal(0.2, 0.2),
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


@memo
def L2[u: UTT, r: OBJ](alpha, p_ident, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(goal in GOALS, wpp=p_ident if goal == 1 else (1.0 - p_ident)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha
                * (
                    goal * log(L1[u, r](alpha, lex, prior) + {EPS})
                    + (1.0 - goal) * log(L0[u, r](lex) + {EPS})
                )
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_salience(ctx, params):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    copy_count = jnp.sum(pair_dist == 0, axis=1)
    is_singleton = jnp.where(copy_count == 1, 1.0, 0.0)
    distinct = jnp.sum(diff, axis=(0, 2))

    salience_logits = (
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
    )
    return softmax_prior(salience_logits)


def choice_probs(params, ctx):
    prior = compute_salience(ctx, params)
    p_ident = jax.nn.sigmoid(params["logit_p_ident"])

    heard = L2(params["alpha"], p_ident, ctx.lex, prior)[ctx.utterance]
    probs = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(probs, params["lapse"])
